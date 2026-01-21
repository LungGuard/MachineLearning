"""
CT Scan Processing Module
==========================
Core logic for processing individual CT scans and extracting nodule data.

This module handles the complete processing pipeline for a single scan:
- Volume loading and preprocessing
- Nodule detection and filtering
- 2.5D image generation
- Metadata extraction

Author: LungGuard ML Team
License: Proprietary
"""

import logging
import traceback
from pathlib import Path
from typing import List, Dict
import numpy as np

# Import dataset utilities
import sys
from pathlib import Path as PathLib
sys.path.insert(0, str(PathLib(__file__).parent.parent))

from utils.dataset_utils import (
    resample_volume,
    apply_windowing,
    create_25d_sandwich,
    compute_nodule_bbox_yolo,
    extract_nodule_features,
    get_nodule_slice_indices,
    get_nodule_centroid
)

from .file_io import AtomicSaveResult, atomic_save_image_and_label

logger = logging.getLogger(__name__)


def process_single_scan(
    scan,
    split: str,
    config,
    directories: Dict[str, Path],
    pl_module
) -> List[Dict]:
    """
    Process a single CT scan and extract nodule data.
    Updated to handle scalar pixel_spacing and ensure robustness.
    
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
        PyLIDC module reference (unused but kept for consistency)
    
    Returns
    -------
    List[Dict]
        List of metadata dictionaries for successfully saved nodules
    """
    metadata_rows = []
    
    try:
        # Load and preprocess the volume
        # Note: This operation can be memory intensive
        volume = scan.to_volume()
        
        # Handle pixel_spacing edge cases where it might be returned as a float
        # instead of a list/tuple (common issue in some pylidc versions/datasets)
        spacing = scan.pixel_spacing
        if isinstance(spacing, (float, int, np.floating, np.integer)):
            spacing = [float(spacing), float(spacing)]
            
        slice_spacing = float(scan.slice_spacing)
        
        # Construct original spacing tuple (z, y, x)
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
                config.min_nodule_diameter <= features['diameter_mm'] <= config.max_nodule_diameter
            )
            
            # Skip invalid nodules without using continue
            process_nodule = diameter_valid and features['annotation_count'] > 0
            
            nodule_results = []
            
            # Get centroid for this nodule cluster
            centroid = get_nodule_centroid(annotations, volume_shape) if process_nodule else None
            
            # Get slice indices containing the nodule
            slice_indices = get_nodule_slice_indices(annotations, volume_shape[0]) if process_nodule else []
            
            # Select representative slices (center + neighbors)
            selected_slices = select_representative_slices(
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
                    features['diameter_mm'],
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
                if metadata_entry:
                    nodule_results.append(metadata_entry)
            
            metadata_rows.extend([r for r in nodule_results if r is not None])
            
    except Exception as e:
        # Improved error logging to catch exactly where it fails
        logger.error(f"Error processing scan {scan.patient_id}: {e}")
        logger.debug(traceback.format_exc())
    
    return metadata_rows


def select_representative_slices(
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
    if total_slices <= num_slices:
        return slice_indices
    elif num_slices > 1:
        return [
            slice_indices[int(i * (total_slices - 1) / (num_slices - 1))]
            for i in range(num_slices)
        ]
    else:
        return [slice_indices[total_slices // 2]]
