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
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json

import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

# Import local utilities
from utils.dataset_utils import (
    resample_volume,
    apply_windowing,
    create_25d_sandwich,
    compute_nodule_bbox_yolo,
    extract_nodule_features,
    get_nodule_slice_indices,
    get_nodule_centroid
)

from constants.detection.dataset_constants import RegModelConstants

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
# Configuration Dataclass
# ==============================================================================

@dataclass
class DataPrepConfig:
    """Configuration for data preparation pipeline."""
    
    # Paths
    data_path: str = "/data/LIDC-IDRI"
    output_dir: str = "./lungguard_dataset"
    
    # Split ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Image parameters
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    window_center: float = -600.0
    window_width: float = 1500.0
    
    # YOLO parameters
    bbox_padding_factor: float = 1.5
    class_id: int = 0  # 0 = nodule/anomaly
    
    # Processing parameters
    min_nodule_diameter: float = 3.0  # mm, minimum nodule size to include
    max_nodule_diameter: float = 100.0  # mm, maximum nodule size
    slices_per_nodule: int = 3  # Number of slices to generate per nodule
    
    # Random seed for reproducibility
    random_seed: int = 42


# ==============================================================================
# PyLIDC Configuration Setup
# ==============================================================================

def configure_pylidc(dicom_path: str) -> bool:
    """
    Dynamically configure pylidc to use a custom DICOM directory.
    
    PyLIDC looks for a configuration file at ~/.pylidcrc
    This function creates/updates that file programmatically.
    
    Parameters
    ----------
    dicom_path : str
        Path to the LIDC-IDRI DICOM data directory
    
    Returns
    -------
    bool
        True if configuration was successful
    
    Notes
    -----
    The pylidc configuration file format:
        [dicom]
        path = /path/to/LIDC-IDRI
    """
    config_path = Path.home() / ".pylidcrc"
    
    # Create configuration content
    config = configparser.ConfigParser()
    config['dicom'] = {'path': str(dicom_path)}
    
    # Write configuration file
    try:
        with open(config_path, 'w') as f:
            config.write(f)
        logger.info(f"PyLIDC configured with DICOM path: {dicom_path}")
        
        # Also set environment variable as backup
        os.environ['PYLIDC_DICOM_PATH'] = str(dicom_path)
        
        success = True
    except Exception as e:
        logger.error(f"Failed to configure pylidc: {e}")
        success = False
    
    return success


def import_pylidc():
    """
    Import pylidc after configuration.
    
    Returns
    -------
    module
        The pylidc module
    """
    import pylidc as pl
    return pl


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
# Patient Split Logic
# ==============================================================================

def split_patients_by_id(
    patient_ids: List[str],
    config: DataPrepConfig
) -> Dict[str, List[str]]:
    """
    Split patient IDs into train/val/test sets.
    
    CRITICAL: Split by patient ID to prevent data leakage.
    A patient's data must exist in ONLY one split.
    
    Parameters
    ----------
    patient_ids : List[str]
        List of unique patient identifiers
    config : DataPrepConfig
        Configuration with split ratios
    
    Returns
    -------
    Dict[str, List[str]]
        Dictionary with 'train', 'val', 'test' keys mapping to patient IDs
    """
    # First split: train vs (val + test)
    train_ids, temp_ids = train_test_split(
        patient_ids,
        train_size=config.train_ratio,
        random_state=config.random_seed
    )
    
    # Second split: val vs test (from remaining)
    val_ratio_adjusted = config.val_ratio / (config.val_ratio + config.test_ratio)
    val_ids, test_ids = train_test_split(
        temp_ids,
        train_size=val_ratio_adjusted,
        random_state=config.random_seed
    )
    
    splits = {
        'train': list(train_ids),
        'val': list(val_ids),
        'test': list(test_ids)
    }
    
    logger.info(
        f"Patient split - Train: {len(train_ids)}, "
        f"Val: {len(val_ids)}, Test: {len(test_ids)}"
    )
    
    return splits


def get_patient_split(patient_id: str, splits: Dict[str, List[str]]) -> str:
    """
    Determine which split a patient belongs to.
    
    Parameters
    ----------
    patient_id : str
        Patient identifier
    splits : Dict[str, List[str]]
        Split dictionary from split_patients_by_id
    
    Returns
    -------
    str
        Split name ('train', 'val', or 'test')
    """
    split_mapping = {
        pid: split_name
        for split_name, pids in splits.items()
        for pid in pids
    }
    
    result = split_mapping.get(patient_id, 'train')  # Default to train if not found
    
    return result


# ==============================================================================
# Atomic Save Operations
# ==============================================================================

@dataclass
class AtomicSaveResult:
    """Result of an atomic save operation."""
    success: bool
    image_path: Optional[str] = None
    label_path: Optional[str] = None
    error_message: Optional[str] = None


def atomic_save_image_and_label(
    image: np.ndarray,
    yolo_bbox: Tuple[float, float, float, float],
    class_id: int,
    image_path: Path,
    label_path: Path
) -> AtomicSaveResult:
    """
    Atomically save image and corresponding YOLO label.
    
    CRITICAL: Label is saved ONLY if image save succeeds.
    This ensures 1:1 alignment between images and labels.
    
    Parameters
    ----------
    image : np.ndarray
        2.5D RGB image to save
    yolo_bbox : Tuple[float, float, float, float]
        YOLO format bbox (x_center, y_center, width, height)
    class_id : int
        Object class ID for YOLO
    image_path : Path
        Destination path for image
    label_path : Path
        Destination path for label
    
    Returns
    -------
    AtomicSaveResult
        Result indicating success/failure and paths
    """
    result = AtomicSaveResult(success=False)
    
    # Attempt image save
    try:
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        save_success = cv2.imwrite(str(image_path), image_bgr)
        
        image_saved = save_success and image_path.exists()
    except Exception as e:
        logger.error(f"Image save failed: {e}")
        image_saved = False
    
    # Save label ONLY if image save succeeded
    try:
        label_content = (
            f"{class_id} "
            f"{yolo_bbox[0]:.6f} "
            f"{yolo_bbox[1]:.6f} "
            f"{yolo_bbox[2]:.6f} "
            f"{yolo_bbox[3]:.6f}\n"
        )
        
        # Write label file only if image was saved
        write_label = image_saved
        
        label_saved = False
        with open(label_path, 'w') as f:
            f.write(label_content) if write_label else None
            label_saved = write_label
            
    except Exception as e:
        logger.error(f"Label save failed: {e}")
        label_saved = False
        # Rollback: delete image if label failed
        image_path.unlink() if image_saved and image_path.exists() else None
        image_saved = False
    
    result = AtomicSaveResult(
        success=image_saved and label_saved,
        image_path=str(image_path) if image_saved else None,
        label_path=str(label_path) if label_saved else None,
        error_message=None if (image_saved and label_saved) else "Save operation failed"
    )
    
    return result


# ==============================================================================
# Scan Processing
# ==============================================================================

def process_single_scan(
    scan,
    split: str,
    config: DataPrepConfig,
    directories: Dict[str, Path],
    pl_module
) -> List[Dict]:
    """
    Process a single CT scan and extract nodule data.
    
    Parameters
    ----------
    scan : pylidc.Scan
        LIDC scan object
    split : str
        Data split ('train', 'val', 'test')
    config : DataPrepConfig
        Processing configuration
    directories : Dict[str, Path]
        Output directory paths
    pl_module : module
        PyLIDC module reference
    
    Returns
    -------
    List[Dict]
        List of metadata dictionaries for successfully saved nodules
    """
    metadata_rows = []
    
    try:
        # Load and preprocess the volume
        volume = scan.to_volume()
        spacing = scan.pixel_spacing  # (row_spacing, col_spacing)
        slice_spacing = scan.slice_spacing
        
        original_spacing = (slice_spacing, spacing[0], spacing[1])
        
        # Resample to isotropic
        resampled = resample_volume(volume, original_spacing, config.target_spacing)
        
        # Apply lung windowing
        windowed = apply_windowing(
            resampled, 
            config.window_center, 
            config.window_width
        )
        
        volume_shape = windowed.shape
        patient_id = scan.patient_id
        
        # Get clustered nodules (nodules agreed upon by multiple radiologists)
        nodules = scan.cluster_annotations()
        
        logger.debug(f"Processing {patient_id}: {len(nodules)} nodule clusters")
        
        # Process each nodule cluster
        for nodule_idx, annotations in enumerate(nodules):
            # Extract features from annotations
            features = extract_nodule_features(annotations)
            
            # Filter by diameter
            diameter_valid = (
                config.min_nodule_diameter <= features[RegModelConstants.Features.FEATURE_DIAMETER_MM] <= config.max_nodule_diameter
            )
            
            # Skip invalid nodules without using continue
            process_nodule = diameter_valid and features[RegModelConstants.Features.FEATURE_ANNOTATION_COUNT] > 0
            
            nodule_results = []
            
            # Get centroid for this nodule cluster
            centroid = get_nodule_centroid(annotations, volume_shape) if process_nodule else None
            
            # Get slice indices containing the nodule
            slice_indices = get_nodule_slice_indices(annotations, volume_shape[0]) if process_nodule else []
            
            # Select representative slices (center + neighbors)
            selected_slices = _select_representative_slices(
                slice_indices, 
                config.slices_per_nodule
            ) if slice_indices else []
            
            # Process each selected slice
            for slice_idx in selected_slices:
                # Create 2.5D image
                image_25d = create_25d_sandwich(windowed, slice_idx)
                
                # Compute YOLO bounding box
                bbox = compute_nodule_bbox_yolo(
                    centroid,
                    features[RegModelConstants.Features.FEATURE_DIAMETER_MM],
                    volume_shape,
                    config.target_spacing,
                    config.bbox_padding_factor
                ) if centroid else None
                
                # Generate unique filename
                filename = f"{patient_id}_n{nodule_idx:02d}_z{slice_idx:04d}"
                
                # Determine output paths based on split
                image_dir = directories[f'{split}_images']
                label_dir = directories[f'{split}_labels']
                
                image_path = image_dir / f"{filename}.jpg"
                label_path = label_dir / f"{filename}.txt"
                
                # Atomic save
                save_result = atomic_save_image_and_label(
                    image_25d,
                    bbox,
                    config.class_id,
                    image_path,
                    label_path
                ) if bbox else AtomicSaveResult(success=False)
                
                # Create metadata row ONLY if save succeeded
                metadata_entry = {
                    'filename': filename,
                    'patient_id': patient_id,
                    'split_group': split,
                    'nodule_index': nodule_idx,
                    'slice_index': slice_idx,
                    'diameter_mm': features['diameter_mm'],
                    'malignancy_score': features['malignancy'],
                    'spiculation': features['spiculation'],
                    'lobulation': features['lobulation'],
                    'subtlety': features['subtlety'],
                    'sphericity': features['sphericity'],
                    'margin': features['margin'],
                    'texture': features['texture'],
                    'calcification': features['calcification'],
                    'internal_structure': features['internal_structure'],
                    'annotation_count': features['annotation_count'],
                    'centroid_z': centroid[0] if centroid else None,
                    'centroid_y': centroid[1] if centroid else None,
                    'centroid_x': centroid[2] if centroid else None,
                    'bbox_x': bbox[0] if bbox else None,
                    'bbox_y': bbox[1] if bbox else None,
                    'bbox_w': bbox[2] if bbox else None,
                    'bbox_h': bbox[3] if bbox else None,
                    'image_path': save_result.image_path,
                    'label_path': save_result.label_path,
                    'volume_depth': volume_shape[0],
                    'volume_height': volume_shape[1],
                    'volume_width': volume_shape[2]
                } if save_result.success else None
                
                # Append only successful entries
                nodule_results.append(metadata_entry) if metadata_entry else None
            
            metadata_rows.extend([r for r in nodule_results if r is not None])
            
    except Exception as e:
        logger.error(f"Error processing scan: {e}")
    
    return metadata_rows


def _select_representative_slices(
    slice_indices: List[int],
    num_slices: int
) -> List[int]:
    """
    Select representative slices from a nodule's slice range.
    
    Strategy: Select center slice and evenly distributed neighbors.
    
    Parameters
    ----------
    slice_indices : List[int]
        All slice indices containing the nodule
    num_slices : int
        Number of slices to select
    
    Returns
    -------
    List[int]
        Selected slice indices
    """
    total_slices = len(slice_indices)
    
    # Handle edge cases
    result = (
        slice_indices
        if total_slices <= num_slices
        else [
            slice_indices[int(i * (total_slices - 1) / (num_slices - 1))]
            for i in range(num_slices)
        ] if num_slices > 1
        else [slice_indices[total_slices // 2]]
    )
    
    return result


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
    splits = split_patients_by_id(patient_ids, config)
    
    # Step 5: Process all scans
    logger.info("Step 5: Processing scans and generating data...")
    all_metadata = []
    
    total_scans = len(scans_with_nodules)
    for idx, scan in enumerate(scans_with_nodules):
        patient_id = scan.patient_id
        split = get_patient_split(patient_id, splits)
        
        # Log progress every 10 scans
        log_progress = (idx + 1) % 10 == 0 or idx == 0 or idx == total_scans - 1
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
    metadata_df = pd.DataFrame(all_metadata)
    
    csv_path = directories['metadata'] / 'regression_dataset.csv'
    metadata_df.to_csv(csv_path, index=False)
    
    # Save configuration for reproducibility
    config_dict = {
        'data_path': config.data_path,
        'output_dir': config.output_dir,
        'train_ratio': config.train_ratio,
        'val_ratio': config.val_ratio,
        'test_ratio': config.test_ratio,
        'target_spacing': config.target_spacing,
        'window_center': config.window_center,
        'window_width': config.window_width,
        'bbox_padding_factor': config.bbox_padding_factor,
        'min_nodule_diameter': config.min_nodule_diameter,
        'max_nodule_diameter': config.max_nodule_diameter,
        'slices_per_nodule': config.slices_per_nodule,
        'random_seed': config.random_seed,
        'generation_timestamp': datetime.now().isoformat(),
        'total_samples': len(metadata_df),
        'train_samples': len(metadata_df[metadata_df['split_group'] == 'train']),
        'val_samples': len(metadata_df[metadata_df['split_group'] == 'val']),
        'test_samples': len(metadata_df[metadata_df['split_group'] == 'test'])
    }
    
    config_path = directories['metadata'] / 'preparation_config.json'
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Generate summary statistics
    logger.info("=" * 60)
    logger.info("Data Preparation Complete!")
    logger.info("=" * 60)
    logger.info(f"Total samples generated: {len(metadata_df)}")
    logger.info(f"  - Train: {config_dict['train_samples']}")
    logger.info(f"  - Val: {config_dict['val_samples']}")
    logger.info(f"  - Test: {config_dict['test_samples']}")
    logger.info(f"Metadata CSV: {csv_path}")
    logger.info(f"Configuration: {config_path}")
    
    # Create YOLO dataset.yaml for training
    yaml_content = f"""# LungGuard YOLO Dataset Configuration
# Auto-generated by prepare_offline_data.py

path: {config.output_dir}
train: train/images
val: val/images
test: test/images

nc: 1  # Number of classes
names:
  0: nodule

# Dataset statistics
# Total images: {len(metadata_df)}
# Train: {config_dict['train_samples']}
# Val: {config_dict['val_samples']}
# Test: {config_dict['test_samples']}
"""
    
    yaml_path = Path(config.output_dir) / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    logger.info(f"YOLO dataset.yaml: {yaml_path}")
    
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