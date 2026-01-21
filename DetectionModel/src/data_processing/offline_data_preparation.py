"""
LungGuard Data Preparation - Offline Data Generator
====================================================
Orchestrates data generation for YOLO detection and regression analysis.

This script:
1. Configures pylidc dynamically for custom DICOM paths
2. Generates 2.5D images for YOLO training
3. Creates aligned metadata CSV for regression
4. Ensures atomic operations and strict data integrity

Author: LungGuard ML Team
License: Proprietary

Usage:
    python prepare_offline_data.py --data_path /path/to/LIDC-IDRI --output_dir /path/to/output
"""

import os
import sys
import argparse
import logging
import configparser
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
import cv2

# Compatibility fixes for older libraries
configparser.SafeConfigParser = configparser.ConfigParser
np.int = np.int64
np.float = np.float64
np.bool = np.bool_
np.object = np.object_
np.str = np.str_

# Import configuration
from .config import DataPrepConfig

# Import pylidc configuration utilities
from .pylidc_config import configure_pylidc, import_pylidc

# Import data splitting utilities
from .data_splitter import split_patients_by_id, get_patient_split


# Import scan processing
from .scan_processor import process_single_scan

# Import dataset writing utilities
from .dataset_writer import (
    save_metadata_csv,
    save_config_json,
    save_yolo_yaml,
    log_summary_statistics
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data_preparation.log')
    ]
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Directory Structure Creation
# ==============================================================================

def create_directory_structure(output_dir: str) -> Dict[str, Path]:
    """
    Create the required directory structure for YOLO training.
    
    Structure:
        output_dir/
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── val/
        │   ├── images/
        │   └── labels/
        ├── test/
        │   ├── images/
        │   └── labels/
        └── metadata/
    
    Parameters
    ----------
    output_dir : str
        Root output directory
    
    Returns
    -------
    Dict[str, Path]
        Dictionary mapping split names to their paths
    """
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
    
    # Create all directories
    for name, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {path}")
    
    logger.info(f"Directory structure created at: {base_path}")
    
    return directories


# ==============================================================================
# Main Processing Pipeline
# ==============================================================================

def run_data_preparation(config: DataPrepConfig) -> Path:
    """
    Execute the full data preparation pipeline.
    
    Parameters
    ----------
    config : DataPrepConfig
        Pipeline configuration
    
    Returns
    -------
    Path
        Path to the generated metadata CSV
    """
    logger.info("=" * 60)
    logger.info("LungGuard Data Preparation Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Configure pylidc
    logger.info("Step 1: Configuring PyLIDC...")
    configure_pylidc(config.data_path)
    pl = import_pylidc()
    
    # Step 2: Create directory structure
    logger.info("Step 2: Creating directory structure...")
    directories = create_directory_structure(config.output_dir)
    
    # Step 3: Query all scans with annotations
    logger.info("Step 3: Querying LIDC database for annotated scans...")
    all_scans = pl.query(pl.Scan).all()
    
    # Filter scans with nodule annotations
    scans_with_nodules = [
        scan for scan in all_scans
        if len(scan.cluster_annotations()) > 0
    ]
    
    logger.info(f"Found {len(scans_with_nodules)} scans with nodule annotations")
    
    # Step 4: Get unique patient IDs and split
    logger.info("Step 4: Splitting patients into train/val/test...")
    patient_ids = list(set(scan.patient_id for scan in scans_with_nodules))
    splits = split_patients_by_id(
        patient_ids,
        config.train_ratio,
        config.val_ratio,
        config.test_ratio,
        config.random_seed
    )
    
    # Step 5: Process all scans
    logger.info("Step 5: Processing scans and generating data...")
    all_metadata = []
    
    total_scans = len(scans_with_nodules)
    for idx, scan in enumerate(scans_with_nodules):
        patient_id = scan.patient_id
        split = get_patient_split(patient_id, splits)
        
        # Log progress according to the config log freq
        log_progress = (idx + 1) % config.log_freq == 0 or idx == 0 or idx == total_scans - 1
        logger.info(
            f"Processing scan {idx + 1}/{total_scans}: {patient_id} ({split})"
        ) if log_progress else None
        
        # Process scan and collect metadata
        scan_metadata = process_single_scan(
            scan, split, config, directories, pl
        )
        all_metadata.extend(scan_metadata)
    
    # Step 6: Create and save metadata CSV
    logger.info("Step 6: Saving metadata CSV...")
    csv_path = directories['metadata'] / 'regression_dataset.csv'
    metadata_df = save_metadata_csv(all_metadata, csv_path)
    
    # Save configuration for reproducibility
    config_path = directories['metadata'] / 'preparation_config.json'
    config_dict = save_config_json(config, config_path, metadata_df)
    
    # Create YOLO dataset.yaml for training
    yaml_path = save_yolo_yaml(config.output_dir, metadata_df)
    
    # Generate summary statistics
    log_summary_statistics(metadata_df, config_dict, csv_path, config_path, yaml_path)
    
    return csv_path


# ==============================================================================
# Entry Point
# ==============================================================================

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='LungGuard Data Preparation Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        default='/data/LIDC-IDRI',
        help='Path to LIDC-IDRI DICOM data'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./lungguard_dataset',
        help='Output directory for generated data'
    )
    
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.70,
        help='Training set ratio'
    )
    
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='Validation set ratio'
    )
    
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.15,
        help='Test set ratio'
    )
    
    parser.add_argument(
        '--min_diameter',
        type=float,
        default=3.0,
        help='Minimum nodule diameter (mm)'
    )
    
    parser.add_argument(
        '--max_diameter',
        type=float,
        default=100.0,
        help='Maximum nodule diameter (mm)'
    )
    
    parser.add_argument(
        '--slices_per_nodule',
        type=int,
        default=3,
        help='Number of slices to generate per nodule'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    logger.setLevel(logging.DEBUG) if args.debug else None
    
    # Create configuration
    config = DataPrepConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_nodule_diameter=args.min_diameter,
        max_nodule_diameter=args.max_diameter,
        slices_per_nodule=args.slices_per_nodule,
        random_seed=args.seed
    )
    
    # Run pipeline
    csv_path = run_data_preparation(config)
    
    return csv_path


# ==============================================================================
# Script Execution
# ==============================================================================

result_path = main() if __name__ == "__main__" else None