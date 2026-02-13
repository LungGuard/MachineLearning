"""
Parallel data preparation pipeline using multiprocessing.

Uses the shared DataPreparationPipeline for setup/finalisation.
Adds a LiveDashboard with pause/resume/abort support and worker
warning suppression so Rich output stays clean.
"""

# ── Compatibility patches MUST come first ──
from pylidc_compat import apply_patches as _apply_compat
_apply_compat()

import logging
import os
import time
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

from .batch_preparation import (
    DataPreparationPipeline,
    DataPrepConfig,
    configure_pylidc,
    get_patient_split,
    import_pylidc,
)
from .scan_processor import CTScanProcessor
from ..sources.scan_adapters import PyLIDCScanSource
from ...utils import NoduleAnnotationProcessor
from terminal_ui import (
    PipelineMode,
    print_completion_banner,
    print_info,
    print_processing_stats,
    print_section_divider,
    print_warning,
)
from pipeline_wizard import LiveDashboard, run_interactive_cleanup
from ..utils.dataset_diagnostics import DatasetDiagnoser

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Worker helpers
# ──────────────────────────────────────────────────────────

_worker_pylidc = None


def _init_worker(data_path: str) -> None:
    """
    Initialize each worker process:
      1. Apply compatibility patches (configparser + numpy)
      2. Suppress ALL warnings so they don't corrupt Rich output
      3. Route logs exclusively to file (no console writes)
      4. Configure its own pylidc connection
    """
    global _worker_pylidc

    # ── Step 1: Compat patches INSIDE the worker process ──
    import configparser
    import numpy as np
    if not hasattr(configparser, "SafeConfigParser"):
        configparser.SafeConfigParser = configparser.ConfigParser
    for alias, repl in {"int": np.int64, "float": np.float64, "bool": np.bool_,
                         "object": np.object_, "str": np.str_}.items():
        if not hasattr(np, alias):
            setattr(np, alias, repl)

    # ── Step 2: Suppress warnings ──
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    warnings.filterwarnings("ignore")

    # ── Step 3: Worker logs → file only ──
    root = logging.getLogger()
    root.handlers.clear()
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(log_dir / "lungguard_pipeline.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | WORKER | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(fh)
    root.setLevel(logging.DEBUG)

    # ── Step 4: Configure pylidc ──
    try:
        configure_pylidc(data_path)
        _worker_pylidc = import_pylidc()
    except Exception as exc:
        logger.error(f"Worker initialisation failed: {exc}")
        _worker_pylidc = None


def _worker_process_scan(args: Tuple) -> List[Dict]:
    """Worker function — process a single scan, return metadata list."""
    patient_id, patient_split, config_dict, directories_dict = args
    global _worker_pylidc
    pylidc = _worker_pylidc

    result: List[Dict] = []
    config = DataPrepConfig(**config_dict)
    directories = {k: Path(v) for k, v in directories_dict.items()}

    try:
        scan = (
            pylidc.query(pylidc.Scan)
            .filter(pylidc.Scan.patient_id == patient_id)
            .first()
        )
        if scan is not None:
            processor = CTScanProcessor(config, directories)
            source = PyLIDCScanSource(scan, NoduleAnnotationProcessor)
            result = processor.process_scan(source, patient_split)
    except Exception as exc:
        logger.error(f"[{patient_id}] Worker error: {exc}")

    return result


# ──────────────────────────────────────────────────────────
# Parallel Orchestrator
# ──────────────────────────────────────────────────────────


def run_parallel_pipeline(
    config: DataPrepConfig,
    num_workers: Optional[int] = None,
) -> Path:
    """
    Orchestrate parallel processing with a LiveDashboard.
    Keyboard controls: [P] Pause  [R] Resume  [Q] Abort
    """

    # 1. Setup
    pipeline = DataPreparationPipeline(config)
    num_workers = num_workers or max(1, cpu_count() - 2)
    pipeline.setup(mode=PipelineMode.PARALLEL, num_workers=num_workers)

    # 2. Prepare task arguments
    config_dict = {
        k: v for k, v in config.__dict__.items() if not k.startswith("_")
    }
    directories_dict = {k: str(v) for k, v in pipeline.directories.items()}

    task_args: List[Tuple] = []
    for scan, _ in pipeline.scans_to_process:
        pid = scan.patient_id
        split = get_patient_split(pid, pipeline.splits)
        task_args.append((pid, split, config_dict, directories_dict))

    total = len(task_args)
    print_section_divider("Parallel Processing")
    print_info(
        f"Dispatching [metric]{total}[/metric] scans across "
        f"[highlight]{num_workers}[/highlight] workers"
    )

    # 3. Live dashboard + multiprocessing
    all_metadata: List[Dict] = []
    dashboard = LiveDashboard(total)
    dashboard.start()
    was_aborted = False

    try:
        with Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(config.data_path,),
        ) as pool:

            results_iter = pool.imap_unordered(
                _worker_process_scan, task_args, chunksize=1,
            )

            for result in results_iter:
                dashboard.poll_commands()

                if dashboard.aborted:
                    print_warning("Aborted — terminating workers…")
                    pool.terminate()
                    pool.join()
                    was_aborted = True
                    all_metadata.clear()

                if not was_aborted:
                    dashboard.wait_while_paused()

                    if result:
                        all_metadata.extend(result)
                        dashboard.advance(scan_images=len(result))
                    else:
                        dashboard.advance(was_error=True)

    except (KeyboardInterrupt, StopIteration):
        print_warning("Interrupted — stopping workers…")
        was_aborted = True
    finally:
        dashboard.stop()

    print_processing_stats(
        total_scans=total,
        successful=dashboard.successful,
        failed=dashboard.failed,
        total_images=dashboard.images,
        elapsed_seconds=dashboard.elapsed,
    )

    # 4. Finalize + cleanup
    extra = {"mode": "parallel", "num_workers": num_workers}
    if was_aborted:
        extra["aborted"] = True

    csv_path = pipeline.finalize(all_metadata, extra)
    run_interactive_cleanup(config.output_dir, DatasetDiagnoser)
    print_completion_banner(log_file=pipeline._log_file)

    return csv_path