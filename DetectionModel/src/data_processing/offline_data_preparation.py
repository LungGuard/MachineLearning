"""Offline data preparation pipeline."""

import os
import sys
import argparse
import logging
import configparser
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# Compatibility fixes
configparser.SafeConfigParser = configparser.ConfigParser
np.int = np.int64
np.float = np.float64
np.bool = np.bool_
np.object = np.object_
np.str = np.str_

logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False) -> None:
    """Configure logging for all modules."""
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
    
    # Create and configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Create and configure file handler
    file_handler = logging.FileHandler('data_preparation.log', mode='w')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger('pylidc').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    # Log confirmation
    root_logger.info(f"Logging initialized at {'DEBUG' if debug else 'INFO'} level")
    root_logger.debug("Debug logging is active - you should see this message")
# Imports after logger setup
from .config import DataPrepConfig
from .pylidc_config import configure_pylidc, import_pylidc
from .data_splitter import split_patients_by_id, get_patient_split
from .scan_processor import process_single_scan
from .dataset_writer import (
    save_metadata_csv,
    save_config_json,
    save_yolo_yaml,
    log_summary_statistics
)


def create_directory_structure(output_dir: str) -> Dict[str, Path]:
    """Create YOLO directory structure."""
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
        path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {base_path}")
    return directories


def filter_scans_with_nodules(all_scans: List, pl) -> List[Tuple]:
    """Filter scans with nodule annotations."""
    total = len(all_scans)
    logger.info(f"Filtering {total} scans for nodule annotations...")
    
    scans_with_nodules = []
    
    for idx, scan in enumerate(all_scans):
        # Progress indicator every 100 scans
        show_progress = (idx + 1) % 100 == 0 or idx == total - 1
        print(f"\r  Checking scan {idx + 1}/{total}...", end='', flush=True) if show_progress else None
        
        try:
            annotations = scan.cluster_annotations()
            has_nodules = len(annotations) > 0
            scans_with_nodules.append((scan, annotations)) if has_nodules else None
        except Exception as e:
            logger.debug(f"Error checking {scan.patient_id}: {e}")
    
    print()  # New line after progress
    logger.info(f"Found {len(scans_with_nodules)} scans with nodules")
    return scans_with_nodules


def run_data_preparation(config: DataPrepConfig) -> Path:
    """Execute the data preparation pipeline."""
    
    logger.info("=" * 50)
    logger.info("LungGuard Data Preparation Pipeline")
    logger.info("=" * 50)
    
    # Step 1: Configure pylidc
    logger.info("[1/6] Configuring PyLIDC...")
    configure_pylidc(config.data_path)
    pl = import_pylidc()
    
    # Step 2: Create directories
    logger.info("[2/6] Creating directories...")
    directories = create_directory_structure(config.output_dir)
    
    # Step 3: Query and filter scans (OPTIMIZED - single pass)
    logger.info("[3/6] Querying LIDC database...")
    all_scans = pl.query(pl.Scan).all()
    logger.info(f"Total scans in database: {len(all_scans)}")
    
    # Filter scans and keep annotations (avoids calling cluster_annotations twice)
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
    
    # Step 5: Process scans
    logger.info("[5/6] Processing scans...")
    all_metadata = []
    successful = 0
    failed = 0
    total_scans = len(scans_with_annotations)
    
    for idx, (scan, annotations) in enumerate(scans_with_annotations):
        patient_id = scan.patient_id
        split = get_patient_split(patient_id, splits)
        
        # Progress display
        show_progress = (idx + 1) % config.log_freq == 0 or idx == 0 or idx == total_scans - 1
        logger.info(f"  [{idx + 1}/{total_scans}] {patient_id} ({split})") if show_progress else None
        
        try:
            scan_metadata = process_single_scan(
                scan, split, config, directories, pl
            )
            
            samples_generated = len(scan_metadata)
            all_metadata.extend(scan_metadata) if samples_generated > 0 else None
            successful += 1 if samples_generated > 0 else 0
            failed += 1 if samples_generated == 0 else 0
            
            logger.debug(f"    -> {samples_generated} samples") if samples_generated > 0 else None
            
        except Exception as e:
            failed += 1
            logger.error(f"  [{patient_id}] Error: {e}")
            logger.debug(f"Traceback:", exc_info=True)
    
    logger.info(f"Processing complete: {successful} OK, {failed} failed")
    logger.info(f"Total samples: {len(all_metadata)}")
    
    # Step 6: Save outputs
    logger.info("[6/6] Saving outputs...")
    csv_path = directories['metadata'] / 'regression_dataset.csv'
    metadata_df = save_metadata_csv(all_metadata, csv_path)
    
    config_path = directories['metadata'] / 'preparation_config.json'
    config_dict = save_config_json(config, config_path, metadata_df)
    
    yaml_path = save_yolo_yaml(config.output_dir, metadata_df)
    
    log_summary_statistics(metadata_df, config_dict, csv_path, config_path, yaml_path)
    
    return csv_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='LungGuard Data Preparation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data_path', type=str, required=True, help='Path to LIDC-IDRI')
    parser.add_argument('--output_dir', type=str, default='./lungguard_dataset', help='Output directory')
    parser.add_argument('--train_ratio', type=float, default=0.70)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--min_diameter', type=float, default=3.0, help='Min nodule diameter (mm)')
    parser.add_argument('--max_diameter', type=float, default=100.0, help='Max nodule diameter (mm)')
    parser.add_argument('--slices_per_nodule', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--log_freq', type=int, default=10, help='Log every N scans')
    
    args = parser.parse_args()
    
    # Setup logging FIRST
    setup_logging(debug=args.debug)
    
    logger.debug(f"Arguments: {args}") if args.debug else None
    
    config = DataPrepConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_nodule_diameter=args.min_diameter,
        max_nodule_diameter=args.max_diameter,
        slices_per_nodule=args.slices_per_nodule,
        random_seed=args.seed,
        log_freq=args.log_freq
    )
    
    csv_path = run_data_preparation(config)
    return csv_path


result_path = main() if __name__ == "__main__" else None