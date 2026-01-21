"""
CT Scan Processing Module
==========================
Core logic for processing individual CT scans and extracting nodule data.

This module handles the complete processing pipeline for a single scan:
- Volume loading and preprocessing
- Nodule detection and filtering
- 2.5D image generation
- Metadata extraction

"""

import logging
import traceback
from pathlib import Path
from typing import List, Dict
import numpy as np
from constants.detection.dataset_constants import RegModelConstants
# Import dataset utilities
import sys
from pathlib import Path as PathLib
sys.path.insert(0, str(PathLib(__file__).parent.parent))

from ..utils.dataset_utils import (
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
        
        logger.debug(f"Processing {patient_id} : {len(nodules)} nodule clusters")
        
        # Define filtering functions
        def is_valid_nodule(nodule_data):
            """Check if nodule meets size and annotation requirements."""
            annotations, features = nodule_data
            diameter_valid = (
                config.min_nodule_diameter <= features[RegModelConstants.Features.FEATURE_DIAMETER_MM] <= config.max_nodule_diameter
            )
            annot_valid = features[RegModelConstants.Features.FEATURE_ANNOTATION_COUNT] > 0
            return diameter_valid and annot_valid
        
        # Extract features for all nodules first
        nodule_features = [
            (annotations, extract_nodule_features(annotations)) 
            for annotations in nodules
        ]
        
        # Filter to only valid nodules
        valid_nodules = list(filter(is_valid_nodule, nodule_features))
        
        logger.debug(f"Valid nodules after filtering: {len(valid_nodules)}/{len(nodules)}")
        
        # Process each valid nodule cluster
        for nodule_idx, (annotations, features) in enumerate(valid_nodules):
            nodule_results = []
            
            # Get centroid for this nodule cluster
            centroid = get_nodule_centroid(annotations, volume_shape)
            
            # Get slice indices containing the nodule
            slice_indices = get_nodule_slice_indices(annotations, volume_shape[0])
            
            # Select representative slices (center + neighbors)
            selected_slices = select_representative_slices(
                slice_indices, 
                config.slices_per_nodule
            )
            
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
                )
                
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
                )
                
                # Create metadata row ONLY if save succeeded
                if save_result.success:
                    metadata_entry = {
                        RegModelConstants.FILE_NAME : filename,
                        RegModelConstants.PATIENT_ID : patient_id,
                        RegModelConstants.SPLIT_GROUP : split,
                        RegModelConstants.NOUDLE_INDEX : nodule_idx,
                        RegModelConstants.SLICE_INDEX : slice_idx,
                        RegModelConstants.Features.FEATURE_DIAMETER_MM : features[RegModelConstants.Features.FEATURE_DIAMETER_MM],
                        f'{RegModelConstants.Features.FEATURE_MALIGNANCY}_score' : features[RegModelConstants.Features.FEATURE_MALIGNANCY],
                        RegModelConstants.Features.FEATURE_SPICULATION : features[RegModelConstants.Features.FEATURE_SPICULATION],
                        RegModelConstants.Features.FEATURE_LOBULATION : features[RegModelConstants.Features.FEATURE_LOBULATION],
                        RegModelConstants.Features.FEATURE_SUBTLETY : features[RegModelConstants.Features.FEATURE_SUBTLETY],
                        RegModelConstants.Features.FEATURE_SPHERICITY : features[RegModelConstants.Features.FEATURE_SPHERICITY],
                        RegModelConstants.Features.FEATURE_MARGIN: features[RegModelConstants.Features.FEATURE_MARGIN],
                        RegModelConstants.Features.FEATURE_TEXTURE : features[RegModelConstants.Features.FEATURE_TEXTURE],
                        RegModelConstants.Features.FEATURE_CALCIFICATION : features[RegModelConstants.Features.FEATURE_CALCIFICATION],
                        RegModelConstants.Features.FEATURE_INTERNAL_STRUCTURE : features[RegModelConstants.Features.FEATURE_INTERNAL_STRUCTURE],
                        RegModelConstants.Features.FEATURE_ANNOTATION_COUNT : features[RegModelConstants.Features.FEATURE_ANNOTATION_COUNT],
                        RegModelConstants.CENTROID.CENTROID_Z : centroid[0],
                        RegModelConstants.CENTROID.CENTROID_Y : centroid[1],
                        RegModelConstants.CENTROID.CENTROID_X : centroid[2],
                        RegModelConstants.BBOX.BBOX_X : bbox[0],
                        RegModelConstants.BBOX.BBOX_Y : bbox[1],
                        RegModelConstants.BBOX.BBOX_W : bbox[2],
                        RegModelConstants.BBOX.BBOX_H : bbox[3],
                        RegModelConstants.IMAGE_PATH : save_result.image_path,
                        RegModelConstants.LABEL_PATH : save_result.label_path,
                        RegModelConstants.VOLUME.VOLUME_DEPTH : volume_shape[0],
                        RegModelConstants.VOLUME.VOLUME_HEIGHT : volume_shape[1],
                        RegModelConstants.VOLUME.VOLUME_WIDTH: volume_shape[2]
                    }
                    nodule_results.append(metadata_entry)
            
            metadata_rows.extend(nodule_results)
            
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
