"""
Unified Data Preparation Pipeline.
Shared setup, configuration, serial processing, and finalization.

The interactive wizard (pipeline_wizard.py) drives configuration;
this module owns the data-processing logic.
"""


import contextlib
from pylidc_compat import apply_patches as _apply_compat
_apply_compat()

import json
import logging
import os
import shutil
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import pandas as pd

from ..config import DataPrepConfig
from ..core.pylidc_config import configure_pylidc, import_pylidc
from ..utils.patient_splitter import split_patients_by_id, get_patient_split
from .scan_processor import CTScanProcessor
from ..sources.scan_adapters import PyLIDCScanSource
from ..sources.annotation_processor import NoduleAnnotationProcessor
from ..io.dataset_writer import (
    save_metadata_csv,
    save_config_json,
    save_yolo_yaml,
    log_summary_statistics,
)
from ..utils.dataset_diagnostics import DatasetDiagnoser
from terminal_ui import (
    PipelineMode,
    console,
    create_filter_progress,
    print_completion_banner,
    print_finalization_summary,
    print_info,
    print_phase_header,
    print_pipeline_banner,
    print_processing_stats,
    print_section_divider,
    print_split_summary,
    print_success,
    print_warning,
    setup_rich_logging,
)
from pipeline_wizard import LiveDashboard, run_interactive_cleanup
from constants.common.model_stages import ModelStage


logger = logging.getLogger(__name__)

TOTAL_SETUP_PHASES = 4


# ──────────────────────────────────────────────────────────
# Directory Structure
# ──────────────────────────────────────────────────────────


def create_directory_structure(output_dir: str) -> Dict[str, Path]:
    """Creates the YOLO directory structure."""
    base_path = Path(output_dir)
    
    directories = {}
    DIR_TYPES= ("images","labels")
    
    for stage in ModelStage:
        for dir_type in DIR_TYPES:
            directories[f'{stage.prefix}{dir_type}'] = base_path / stage.value / dir_type
    
    directories["metadata"] = base_path / "metadata"
    

    for name, path in directories.items():
        is_metadata = name == "metadata"
        if path.exists() and not is_metadata:
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        if is_metadata:
            for f in path.iterdir():
                if f.is_file() and not f.name.startswith("patient_splits"):
                    f.unlink()

    return directories


def filter_scans_with_nodules(all_scans: List, pylidc_module) -> List[Tuple]:
    """Filters scans that contain clustered annotations, with progress bar."""
    total = len(all_scans)
    scans_with_nodules: List[Tuple] = []

    with create_filter_progress() as progress:
        task = progress.add_task("Filtering scans for nodules", total=total)
        for scan in all_scans:
            with contextlib.suppress(Exception):
                clusters = scan.cluster_annotations()
                if len(clusters) > 0:
                    scans_with_nodules.append((scan, clusters))
            progress.advance(task)

    print_success(
        f"Found [metric]{len(scans_with_nodules)}[/metric] scans "
        f"with nodules (out of {total})"
    )
    return scans_with_nodules


# ──────────────────────────────────────────────────────────
# Pipeline Core
# ──────────────────────────────────────────────────────────


class DataPreparationPipeline:
    """
    Handles setup, patient splitting, serial processing with
    a live dashboard (pause/resume/skip/abort), and finalization.
    """

    def __init__(self, config: DataPrepConfig):
        self.config = config
        self.pylidc = None
        self.directories: Dict[str, Path] = {}
        self.scans_to_process: List[Tuple] = []
        self.splits: Dict[str, list] = {}
        self._log_file: Optional[Path] = None

    # ── Setup ────────────────────────────────────────────

    def setup(
        self,
        mode: PipelineMode = PipelineMode.SERIAL,
        num_workers: Optional[int] = None,
    ) -> None:
        self._log_file = setup_rich_logging(debug=getattr(self.config, "debug", False))
        print_pipeline_banner(mode, num_workers)
        print_section_divider("Setup")

        print_phase_header(1, TOTAL_SETUP_PHASES, "Configuring PyLIDC…")
        configure_pylidc(self.config.data_path)
        self.pylidc = import_pylidc()
        print_success("PyLIDC configured")

        print_phase_header(2, TOTAL_SETUP_PHASES, "Creating directory structure…")
        self.directories = create_directory_structure(self.config.output_dir)
        print_success(f"Directories ready at [muted]{self.config.output_dir}[/muted]")

        self._prepare_splits_and_scans()

    def _prepare_splits_and_scans(self) -> None:
        splits_json = self.directories["metadata"] / "patient_splits.json"
        existing = splits_json.exists()
        label = "Loading existing patient splits…" if existing else "Querying database & creating splits…"
        print_phase_header(3, TOTAL_SETUP_PHASES, label)

        loader = self._load_existing_splits if existing else self._create_new_splits
        loader(splits_json)

        print_phase_header(4, TOTAL_SETUP_PHASES, "Summary")
        print_split_summary(self.splits)
        print_info(f"Ready to process [metric]{len(self.scans_to_process)}[/metric] scans")

    def _load_existing_splits(self, splits_json: Path) -> None:
        with open(splits_json, "r") as f:
            patient_splits_dict = json.load(f)

        self.splits = {"train": [], "val": [], "test": []}
        for pid, split in patient_splits_dict.items():
            self.splits[split].append(pid)

        patient_ids = list(patient_splits_dict.keys())
        with create_filter_progress() as progress:
            task = progress.add_task("Loading patient scans", total=len(patient_ids))
            for pid in patient_ids:
                with contextlib.suppress(Exception):
                    scans = (
                        self.pylidc.query(self.pylidc.Scan)
                        .filter(self.pylidc.Scan.patient_id == pid)
                        .all()
                    )
                    scans = list(filter(lambda scan:scan.cluster_annotations()>0,scans))
                    self.scans_to_process = [(scan,scan.cluster_annotations()) for scan in scans] 

                progress.advance(task)

    def _create_new_splits(self, splits_json: Path) -> None:
        all_scans = self.pylidc.query(self.pylidc.Scan).all()
        self.scans_to_process = filter_scans_with_nodules(all_scans, self.pylidc)

        if not self.scans_to_process:
            raise ValueError("No valid scans found in database!")

        patient_ids = list({s.patient_id for s, _ in self.scans_to_process})
        self.splits = split_patients_by_id(
            patient_ids,
            self.config.train_ratio,
            self.config.val_ratio,
            self.config.test_ratio,
            self.config.random_seed,
        )

        patient_splits_dict = {
            pid: get_patient_split(pid, self.splits) for pid in patient_ids
        }
        with open(splits_json, "w") as f:
            json.dump(patient_splits_dict, f, indent=2, sort_keys=True)
        print_success("Patient splits saved")

    # ── Finalization ─────────────────────────────────────

    def finalize(self, all_metadata: List[Dict], extra_config_info: Optional[Dict] = None) -> Path:
        print_section_divider("Finalization")

        csv_path = self.directories["metadata"] / "regression_dataset.csv"
        metadata_df = save_metadata_csv(all_metadata, csv_path)

        config_path = self.directories["metadata"] / "preparation_config.json"
        config_dict_full = save_config_json(self.config, config_path, metadata_df)
        if extra_config_info:
            config_dict_full.update(extra_config_info)

        yaml_path = save_yolo_yaml(self.config.output_dir, metadata_df)
        log_summary_statistics(metadata_df, config_dict_full, csv_path, config_path, yaml_path)
        print_finalization_summary(csv_path, config_path, yaml_path, len(metadata_df))

        return csv_path

    # ── Serial Processing with Live Dashboard ────────────

    def run_serial(self) -> Path:
        """
        Serial processing loop wrapped in a LiveDashboard.
        Supports pause, resume, skip, and abort via keyboard.
        """
        self.setup(mode=PipelineMode.SERIAL)
        processor = CTScanProcessor(self.config, self.directories)

        all_metadata: List[Dict] = []
        total = len(self.scans_to_process)
        was_aborted = False

        print_section_divider("Processing Scans")
        dashboard = LiveDashboard(total)
        dashboard.start()

        try:
            for scan, _ in self.scans_to_process:
                dashboard.poll_commands()

                was_aborted = dashboard.aborted
                if not was_aborted:
                    dashboard.wait_while_paused()
                    was_aborted = dashboard.aborted

                if was_aborted:
                    print_warning("Aborted by user")
                else:
                    processor = CTScanProcessor(self.config, self.directories)
                    self._process_one_scan(scan, processor, dashboard, all_metadata)
        finally:
            dashboard.stop()

        print_processing_stats(
            total_scans=total,
            successful=dashboard.successful,
            failed=dashboard.failed,
            total_images=dashboard.images,
            elapsed_seconds=dashboard.elapsed,
        )

        extra = {"mode": "serial"}
        if was_aborted:
            extra["aborted"] = True

        csv_path = self.finalize(all_metadata, extra)
        run_interactive_cleanup(self.config.output_dir, DatasetDiagnoser)
        print_completion_banner(log_file=self._log_file)

        return csv_path

    def _process_one_scan(
        self,
        scan,
        processor: CTScanProcessor,
        dashboard: LiveDashboard,
        all_metadata: List[Dict],
    ) -> None:
        """Process a single scan, respecting skip. Mutates all_metadata in place."""
        if dashboard.skip_current:
            dashboard.advance(was_error=False)
            logger.info(f"Skipped {scan.patient_id}")
            dashboard.skip_current = False
            return

        split = get_patient_split(scan.patient_id, self.splits)
        try:
            source = PyLIDCScanSource(scan, NoduleAnnotationProcessor)
            meta = processor.process_scan(source, split)
            all_metadata.extend(meta)
            dashboard.advance(scan_images=len(meta))
        except Exception as e:
            logger.error(f"Error processing {scan.patient_id}: {e}")
            dashboard.advance(was_error=True)