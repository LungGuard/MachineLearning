"""Parallel offline data preparation pipeline using multiprocessing."""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd

# Import reusable components from sequential pipeline
from .offline_data_preparation import (
    DEFAULT_CONFIG,
    setup_logging,
    create_directory_structure,
    filter_scans_with_nodules,
)
from .config import DataPrepConfig
from .pylidc_config import configure_pylidc, import_pylidc
from .data_splitter import split_patients_by_id, get_patient_split
from .scan_processor import CTScanProcessor
from .dataset_writer import (
    save_metadata_csv,
    save_config_json,
    save_yolo_yaml,
    log_summary_statistics
)


logger = logging.getLogger(__name__)


# Extended configuration with parallel processing options
PARALLEL_DEFAULT_CONFIG = {
    **DEFAULT_CONFIG,
    'num_workers': None,  # None = cpu_count() - 1
    'chunk_size': 1,      # Scans per task
}


def _worker_process_scan(args: Tuple) -> Tuple[str, List[Dict], bool]:
    """
    Worker function for multiprocessing - processes a single scan.
    
    This must be a top-level function (not nested) to be picklable.
    
    Args:
        args: Tuple of (scan, annotations, patient_split, config_dict, directories_dict)
        
    Returns:
        Tuple of (patient_id, metadata_list, success)
    """
    scan, annotations, patient_split, config_dict, directories_dict = args
    
    # Reconstruct config and directories from dictionaries
    config = DataPrepConfig(**config_dict)
    directories = {k: Path(v) for k, v in directories_dict.items()}
    
    patient_id = scan.patient_id
    
    # Create processor instance for this worker
    processor = CTScanProcessor(config, directories)
    
    try:
        # Process the scan
        scan_metadata = processor.process_scan(scan, patient_split, pl=None)
        success = len(scan_metadata) > 0
        
        if success:
            logger.debug(f"Worker processed {patient_id}: {len(scan_metadata)} samples")
        
        return patient_id, scan_metadata, success
        
    except Exception as e:
        logger.error(f"[{patient_id}] Worker error: {e}")
        logger.debug(f"Traceback:", exc_info=True)
        return patient_id, [], False


def run_parallel_data_preparation(config: DataPrepConfig, num_workers: int = None) -> Path:
    """
    Execute the data preparation pipeline with parallel processing.
    
    Args:
        config: DataPrepConfig instance
        num_workers: Number of worker processes (default: cpu_count() - 1)
        
    Returns:
        Path to the generated metadata CSV file
    """
    
    logger.info("=" * 50)
    logger.info("LungGuard Parallel Data Preparation Pipeline")
    logger.info("=" * 50)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)
    num_workers = max(1, min(num_workers, cpu_count()))
    
    logger.info(f"Using {num_workers} worker processes (CPU count: {cpu_count()})")
    
    # Step 1: Configure pylidc
    logger.info("[1/6] Configuring PyLIDC...")
    configure_pylidc(config.data_path)
    pl = import_pylidc()
    
    # Step 2: Create directories
    logger.info("[2/6] Creating directories...")
    directories = create_directory_structure(config.output_dir)
    
    # Step 3: Query and filter scans
    logger.info("[3/6] Querying LIDC database...")
    all_scans = pl.query(pl.Scan).all()
    logger.info(f"Total scans in database: {len(all_scans)}")
    
    scans_with_annotations = filter_scans_with_nodules(all_scans, pl)
    
    if len(scans_with_annotations) == 0:
        logger.error("No scans with nodule annotations found!")
        csv_path = directories['metadata'] / 'regression_dataset.csv'
        pd.DataFrame().to_csv(csv_path, index=False)
        return csv_path
    
    # Step 4: Split patients
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
    
    # Step 5: Process scans in parallel
    logger.info("[5/6] Processing scans in parallel...")
    logger.info(f"Total scans to process: {len(scans_with_annotations)}")
    
    # Prepare serializable data for workers
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
    
    # Prepare task arguments for each scan
    task_args = []
    for scan, annotations in scans_with_annotations:
        patient_id = scan.patient_id
        patient_split = get_patient_split(patient_id, splits)
        task_args.append((scan, annotations, patient_split, config_dict, directories_dict))
    
    # Process scans in parallel
    all_metadata = []
    successful = 0
    failed = 0
    
    try:
        # Use 'spawn' method for Windows compatibility
        with Pool(processes=num_workers) as pool:
            # Use imap_unordered for progress tracking
            results = pool.imap_unordered(
                _worker_process_scan,
                task_args,
                chunksize=1
            )
            
            total_scans = len(task_args)
            
            for idx, (patient_id, scan_metadata, success) in enumerate(results):
                # Track results
                if success:
                    successful += 1
                    all_metadata.extend(scan_metadata)
                else:
                    failed += 1
                
                # Progress display
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
    
    # Step 6: Save outputs
    logger.info("[6/6] Saving outputs...")
    csv_path = directories['metadata'] / 'regression_dataset.csv'
    metadata_df = save_metadata_csv(all_metadata, csv_path)
    
    config_path = directories['metadata'] / 'preparation_config.json'
    config_dict_with_workers = {**config_dict, 'num_workers': num_workers}
    config_dict_full = save_config_json(config, config_path, metadata_df)
    config_dict_full['num_workers'] = num_workers
    
    yaml_path = save_yolo_yaml(config.output_dir, metadata_df)
    
    log_summary_statistics(metadata_df, config_dict_full, csv_path, config_path, yaml_path)
    
    return csv_path


def main_parallel(config_overrides: Dict = None, num_workers: int = None):
    """
    Main entry point for parallel data preparation.
    
    Args:
        config_overrides: Dictionary of configuration overrides
        num_workers: Number of worker processes (None = auto-detect)
        
    Returns:
        Path to the generated metadata CSV file
    """
    config_values = PARALLEL_DEFAULT_CONFIG.copy()
    if config_overrides:
        config_values.update(config_overrides)
    
    # Override num_workers if explicitly provided
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
    # Example usage
    config_overrides = {
        #'data_path': '/path/to/LIDC-IDRI',  # REQUIRED: Set your LIDC-IDRI path
        # 'output_dir': './my_custom_output',  # Uncomment to change output directory
        # 'min_diameter': 5.0,  # Uncomment to change minimum nodule diameter
        # 'debug': True,  # Uncomment for debug logging
        # 'num_workers': 4,  # Uncomment to set specific number of workers
    }
    
    result_path = main_parallel(config_overrides)
    print(f"\nDataset preparation complete!")
    print(f"Metadata saved to: {result_path}")
