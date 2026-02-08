"""Parallel offline data preparation pipeline using multiprocessing."""

import os
import sys
import logging
import configparser
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd

configparser.SafeConfigParser = configparser.ConfigParser
np.int = np.int64
np.float = np.float64
np.bool = np.bool_
np.object = np.object_
np.str = np.str_

from .offline_data_preparation import (
    DEFAULT_CONFIG,
    setup_logging,
    create_directory_structure,
    filter_scans_with_nodules,
)
from .config import DataPrepConfig
from .pylidc_config import configure_pylidc, import_pylidc
from .data_splitter import split_patients_by_id, get_patient_split
from .MONAI_scan_processor import CTScanProcessor as MonaiProcessor
from .dataset_writer import (
    save_metadata_csv,
    save_config_json,
    save_yolo_yaml,
    log_summary_statistics
)


logger = logging.getLogger(__name__)


PARALLEL_DEFAULT_CONFIG = {
    **DEFAULT_CONFIG,
    'num_workers': None,
    'chunk_size': 1,
}


_worker_pylidc = None


def _init_worker(data_path: str):
    """Initialize each worker process with its own pylidc configuration."""
    global _worker_pylidc
    try:
        configure_pylidc(data_path)
        _worker_pylidc = import_pylidc()
    except Exception as e:
        logger.error(f"Worker initialization failed: {e}")
        _worker_pylidc = None


def _worker_process_scan(args: Tuple) -> Tuple[str, List[Dict], bool]:
    patient_id, patient_split, config_dict, directories_dict = args

    global _worker_pylidc
    pylidc = _worker_pylidc

    config = DataPrepConfig(**config_dict)
    directories = {k: Path(v) for k, v in directories_dict.items()}

    try:
        scan = pylidc.query(pylidc.Scan).filter(pylidc.Scan.patient_id == patient_id).first()

        if not scan:
            logger.error(f"Worker could not find scan for {patient_id}")
            return patient_id, [], False

        processor = MonaiProcessor(config, directories)

        scan_metadata = processor.process_scan(scan, patient_split, pl_module=pylidc)
        success = len(scan_metadata) > 0

        return patient_id, scan_metadata, success

    except Exception as e:
        logger.error(f"[{patient_id}] Worker error: {e}")
        return patient_id, [], False


def run_parallel_data_preparation(config: DataPrepConfig, num_workers: int = None) -> Path:
    logger.info("=" * 50)
    logger.info("LungGuard Parallel Data Preparation Pipeline")
    logger.info("=" * 50)

    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)
    num_workers = max(1, min(num_workers, cpu_count()))

    logger.info(f"Using {num_workers} worker processes (CPU count: {cpu_count()})")

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
        logger.info(f"Total scans in database: {len(all_scans)}")

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

    logger.info("[5/6] Processing scans in parallel...")
    logger.info(f"Total scans to process: {len(scans_with_annotations)}")

    config_dict = {
        'data_path': config.data_path,
        'output_dir': config.output_dir,
        'train_ratio': config.train_ratio,
        'val_ratio': config.val_ratio,
        'test_ratio': config.test_ratio,
        'min_nodule_diameter': config.min_nodule_diameter,
        'max_nodule_diameter': config.max_nodule_diameter,
        'slices_per_nodule': config.slices_per_nodule,
        'random_seed': config.random_seed,
        'log_freq': config.log_freq
    }

    directories_dict = {k: str(v) for k, v in directories.items()}

    task_args = []
    for scan, annotations in scans_with_annotations:
        patient_id = scan.patient_id
        patient_split = get_patient_split(patient_id, splits)
        task_args.append((patient_id, patient_split, config_dict, directories_dict))

    all_metadata = []
    successful = 0
    failed = 0

    try:
        with Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(config.data_path,)
        ) as pool:
            results = pool.imap_unordered(
                _worker_process_scan,
                task_args,
                chunksize=1
            )

            total_scans = len(task_args)

            for idx, (patient_id, scan_metadata, success) in enumerate(results):
                if success:
                    successful += 1
                    all_metadata.extend(scan_metadata)
                else:
                    failed += 1

                show_progress = (idx + 1) % config.log_freq == 0 or idx == 0 or idx == total_scans - 1
                if show_progress:
                    logger.info(
                        f"  [{idx + 1}/{total_scans}] Completed: {successful} OK, "
                        f"{failed} failed, {len(all_metadata)} samples"
                    )

    except KeyboardInterrupt:
        logger.warning("Interrupted by user! Cleaning up workers...")
        pool.terminate()
        pool.join()
        raise
    except Exception as e:
        logger.error(f"Error during parallel processing: {e}")
        raise

    logger.info(f"Processing complete: {successful} OK, {failed} failed")
    logger.info(f"Total samples: {len(all_metadata)}")

    logger.info("[6/6] Saving outputs...")
    csv_path = directories['metadata'] / 'regression_dataset.csv'
    metadata_df = save_metadata_csv(all_metadata, csv_path)

    config_path = directories['metadata'] / 'preparation_config.json'
    config_dict_full = save_config_json(config, config_path, metadata_df)
    config_dict_full['num_workers'] = num_workers

    yaml_path = save_yolo_yaml(config.output_dir, metadata_df)

    log_summary_statistics(metadata_df, config_dict_full, csv_path, config_path, yaml_path)

    return csv_path


def main_parallel(config_overrides: Dict = None, num_workers: int = None):
    config_values = PARALLEL_DEFAULT_CONFIG.copy()
    if config_overrides:
        config_values.update(config_overrides)

    if num_workers is not None:
        config_values['num_workers'] = num_workers

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

    csv_path = run_parallel_data_preparation(config, num_workers=config_values['num_workers'])
    return csv_path


if __name__ == "__main__":
    config_overrides = {
        'output_dir': r".\DetectionModel\datasets_monai",
    }

    result_path = main_parallel(config_overrides)
    print(f"\nDataset preparation complete!")
    print(f"Metadata saved to: {result_path}")
