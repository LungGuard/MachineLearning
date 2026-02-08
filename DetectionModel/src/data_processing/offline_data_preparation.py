"""Offline data preparation pipeline."""

import os
import sys
import logging
import configparser
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import shutil

configparser.SafeConfigParser = configparser.ConfigParser
np.int = np.int64
np.float = np.float64
np.bool = np.bool_
np.object = np.object_
np.str = np.str_

logger = logging.getLogger(__name__)


DEFAULT_CONFIG = {
    'data_path': r"E:\FinalsProject\Datasets\CancerDetection\images\manifest-1600709154662\LIDC-IDRI",
    'output_dir': r".\DetectionModel\datasets",
    'train_ratio': 0.70,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'min_diameter': 3.0,
    'max_diameter': 100.0,
    'slices_per_nodule': 3,
    'seed': 42,
    'debug': False,
    'log_freq': 5
}


def setup_logging(debug: bool = False) -> None:
    log_level = logging.DEBUG if debug else logging.INFO
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)

    log_format = (
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        if debug
        else '%(asctime)s - %(message)s'
    )
    formatter = logging.Formatter(log_format, datefmt='%H:%M:%S')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler('data_preparation.log', mode='w')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.getLogger('pylidc').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

from .config import DataPrepConfig
from .pylidc_config import configure_pylidc, import_pylidc
from .data_splitter import split_patients_by_id, get_patient_split
from .scan_processor import CTScanProcessor
from .MONAI_scan_processor import CTScanProcessor as MonaiCTProcessor
from .dataset_writer import (
    save_metadata_csv,
    save_config_json,
    save_yolo_yaml,
    log_summary_statistics
)


def create_directory_structure(output_dir: str) -> Dict[str, Path]:
    base_path = Path(output_dir)

    directories = {
        'train_images': base_path / 'train' / 'images',
        'train_labels': base_path / 'train' / 'labels',
        'val_images': base_path / 'val' / 'images',
        'val_labels': base_path / 'val' / 'labels',
        'test_images': base_path / 'test' / 'images',
        'test_labels': base_path / 'test' / 'labels',
        'metadata': base_path / 'metadata'
    }

    for name, path in directories.items():
        cleanup_needed = name != 'metadata'

        path_exists = path.exists()
        should_remove = cleanup_needed and path_exists
        shutil.rmtree(path) if should_remove else None

        is_metadata = name == 'metadata'
        should_clean_files = is_metadata and path_exists

        if should_clean_files:
            for file_path in path.iterdir():
                is_file = file_path.is_file()
                is_patient_splits = file_path.name.startswith("patient_splits")
                should_delete = is_file and not is_patient_splits
                file_path.unlink() if should_delete else None

        path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {base_path}")
    return directories


def filter_scans_with_nodules(all_scans: List, pylidc) -> List[Tuple]:
    total = len(all_scans)
    logger.info(f"Filtering {total} scans for nodule annotations...")

    scans_with_nodules = []

    for idx, scan in enumerate(all_scans):
        show_progress = (idx + 1) % 100 == 0 or idx == total - 1
        logger.info(f"  Checking scan {idx + 1}/{total}...") if show_progress else None

        try:
            annotations = scan.cluster_annotations()
            has_nodules = len(annotations) > 0
            scans_with_nodules.append((scan, annotations)) if has_nodules else None
        except Exception as e:
            logger.debug(f"Error checking {scan.patient_id}: {e}")

    logger.info(f"Found {len(scans_with_nodules)} scans with nodules")
    return scans_with_nodules


def run_data_preparation(config: DataPrepConfig) -> Path:
    logger.info("=" * 50)
    logger.info("LungGuard Data Preparation Pipeline")
    logger.info("=" * 50)

    logger.info("[1/6] Configuring PyLIDC...")
    configure_pylidc(config.data_path)
    pylidc = import_pylidc()

    logger.info("[2/6] Creating directories...")
    directories = create_directory_structure(config.output_dir)

    splits_json_path = directories['metadata'] / 'patient_splits.json'

    if splits_json_path.exists():
        logger.info("[3/6] Loading existing patient splits...")
        with open(splits_json_path, 'r') as f:
            patient_splits_dict = json.load(f)

        patient_ids = list(patient_splits_dict.keys())
        logger.info(f"Loaded {len(patient_ids)} patients from existing splits")

        logger.info("[4/6] Querying scans for existing patients...")
        scans_with_annotations = []
        for patient_id in patient_ids:
            try:
                patient_scans = pylidc.query(pylidc.Scan).filter(pylidc.Scan.patient_id == patient_id).all()
                for scan in patient_scans:
                    annotations = scan.cluster_annotations()
                    if len(annotations) > 0:
                        scans_with_annotations.append((scan, annotations))
            except Exception as e:
                logger.debug(f"Error querying {patient_id}: {e}")

        logger.info(f"Found {len(scans_with_annotations)} scans with nodules")

        splits = {'train': [], 'val': [], 'test': []}
        for patient_id, split in patient_splits_dict.items():
            splits[split].append(patient_id)

    else:
        logger.info("[3/6] Querying LIDC database...")
        all_scans = pylidc.query(pylidc.Scan).all()
        logger.info(f"Total scans found: {len(all_scans)}")

        scans_with_annotations = filter_scans_with_nodules(all_scans, pylidc)

        if len(scans_with_annotations) == 0:
            logger.error("No scans with nodule annotations found!")
            csv_path = directories['metadata'] / 'regression_dataset.csv'
            pd.DataFrame().to_csv(csv_path, index=False)
            return csv_path

        logger.info("[4/6] Splitting patients...")
        patient_ids = list(set(scan.patient_id for scan, _ in scans_with_annotations))
        logger.info(f"Unique patients: {len(patient_ids)}")

        splits = split_patients_by_id(
            patient_ids,
            config.train_ratio,
            config.val_ratio,
            config.test_ratio,
            config.random_seed
        )

        patient_splits_dict = {}
        for patient_id in patient_ids:
            split = get_patient_split(patient_id, splits)
            patient_splits_dict[patient_id] = split

        with open(splits_json_path, 'w') as f:
            json.dump(patient_splits_dict, f, indent=2, sort_keys=True)
        logger.info(f"Patient splits saved to: {splits_json_path}")

    logger.info("[5/6] Processing scans...")

    processor = MonaiCTProcessor(config, directories)

    all_metadata = []
    successful = 0
    failed = 0
    total_scans = len(scans_with_annotations)

    for idx, (scan, annotations) in enumerate(scans_with_annotations):
        patient_id = scan.patient_id
        split = get_patient_split(patient_id, splits)

        show_progress = (idx + 1) % config.log_freq == 0 or idx == 0 or idx == total_scans - 1
        logger.info(f"  [{idx + 1}/{total_scans}] {patient_id} ({split})") if show_progress else None

        try:
            scan_metadata = processor.process_scan(scan, split, pylidc)

            samples_generated = len(scan_metadata)
            all_metadata.extend(scan_metadata) if samples_generated > 0 else None
            successful += 1 if samples_generated > 0 else 0
            failed += 1 if samples_generated == 0 else 0

        except Exception as e:
            failed += 1
            logger.error(f"  [{patient_id}] Error: {e}")
            logger.debug(f"Traceback:", exc_info=True)

    logger.info(f"Processing complete: {successful} OK, {failed} failed")
    logger.info(f"Total samples: {len(all_metadata)}")

    logger.info("[6/6] Saving outputs...")
    csv_path = directories['metadata'] / 'regression_dataset.csv'
    metadata_df = save_metadata_csv(all_metadata, csv_path)

    config_path = directories['metadata'] / 'preparation_config.json'
    config_dict = save_config_json(config, config_path, metadata_df)

    yaml_path = save_yolo_yaml(config.output_dir, metadata_df)

    log_summary_statistics(metadata_df, config_dict, csv_path, config_path, yaml_path)

    return csv_path


def main(config_overrides: Dict = None):
    config_values = DEFAULT_CONFIG.copy()
    if config_overrides:
        config_values.update(config_overrides)

    setup_logging(debug=config_values['debug'])

    if config_values['debug']:
        logger.debug(f"Configuration: {config_values}")

    config = DataPrepConfig(
        data_path=config_values['data_path'],
        output_dir=config_values['output_dir'],
        train_ratio=config_values['train_ratio'],
        val_ratio=config_values['val_ratio'],
        test_ratio=config_values['test_ratio'],
        min_nodule_diameter=config_values['min_diameter'],
        max_nodule_diameter=config_values['max_diameter'],
        slices_per_nodule=config_values['slices_per_nodule'],
        random_seed=config_values['seed'],
        log_freq=config_values['log_freq']
    )

    csv_path = run_data_preparation(config)
    return csv_path


if __name__ == "__main__":
    config_overrides = {
        'output_dir': r".\DetectionModel\datasets_monai",
        'debug': True,
    }

    result_path = main(config_overrides)
