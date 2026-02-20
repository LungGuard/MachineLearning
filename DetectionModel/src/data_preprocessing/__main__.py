"""
LungGuard Data Preparation — Interactive Entry Point.

Run with:
    python -m DetectionModel.src.data_preprocessing

Flow:
  1. Startup Wizard  → collects mode, paths, ratios, workers, advanced opts
  2. Live Dashboard   → real-time progress with [P]ause [R]esume [S]kip [Q]uit
  3. Post-Processing → diagnosis, reports, dataset cleanup
"""

from pylidc_compat import apply_patches as _apply_compat
_apply_compat()

import sys
from pathlib import Path

from .config import DataPrepConfig
from terminal_ui import PipelineMode, print_error
from pipeline_wizard import run_wizard


def main() -> Path:
    """Interactive entry point — wizard → pipeline → cleanup → done."""

    config_dict = run_wizard()

    config = DataPrepConfig(
        data_path=config_dict["data_path"],
        output_dir=config_dict["output_dir"],
        train_ratio=config_dict["train_ratio"],
        val_ratio=config_dict["val_ratio"],
        test_ratio=config_dict["test_ratio"],
        min_nodule_diameter=config_dict["min_nodule_diameter"],
        max_nodule_diameter=config_dict["max_nodule_diameter"],
        slices_per_nodule=config_dict["slices_per_nodule"],
        random_seed=config_dict["random_seed"],
        log_freq=config_dict["log_freq"],
    )

    mode = PipelineMode(config_dict["mode"])
    return _dispatch(mode, config, config_dict.get("num_workers"))


def _dispatch(mode: PipelineMode, config: DataPrepConfig, num_workers=None) -> Path:
    """Route to the correct pipeline based on the wizard's mode selection."""

    result: Path

    if mode == PipelineMode.SERIAL:
        from .pipelines.batch_preparation import DataPreparationPipeline
        pipeline = DataPreparationPipeline(config)
        result = pipeline.run_serial()
    else:
        from .pipelines.parallel_preparation import run_parallel_pipeline
        result = run_parallel_pipeline(config, num_workers=num_workers)

    return result


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_error("Interrupted.")
        sys.exit()