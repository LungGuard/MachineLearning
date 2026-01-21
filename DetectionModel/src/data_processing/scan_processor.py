"""
CT Scan Processing Module 
===============================================
Added detailed logging to identify why all scans are failing.
"""

import logging
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from constants.detection.dataset_constants import RegModelConstants

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


def prepare_scan_volume(
    scan,
    patient_id: str,
    config
) -> Optional[Tuple[np.ndarray, Tuple[int, int, int], Tuple[float, float, float]]]:
    """
    Load, resample, and window a CT scan volume.
    
    Returns
    -------
    Optional[Tuple[np.ndarray, Tuple[int, int, int], Tuple[float, float, float]]]
        Tuple of (windowed_volume, volume_shape, original_spacing) or None if failed
    """
    # Step 1: Load volume
    try:
        logger.debug(f"[{patient_id}] Loading volume...")
        volume = scan.to_volume()
        logger.debug(f"[{patient_id}] Volume loaded: shape={volume.shape}, dtype={volume.dtype}")
    except Exception as e:
        logger.error(f"[{patient_id}] FAILED at volume loading: {e}")
        return None
    
    # Step 2: Get spacing
    try:
        spacing = scan.pixel_spacing
        logger.debug(f"[{patient_id}] Raw pixel_spacing: {spacing} (type={type(spacing)})")
        
        # Handle scalar spacing
        spacing = (
            [float(spacing), float(spacing)]
            if isinstance(spacing, (float, int, np.floating, np.integer))
            else [float(spacing[0]), float(spacing[1])]
        )
        
        slice_spacing = float(scan.slice_spacing)
        original_spacing = (slice_spacing, spacing[0], spacing[1])
        logger.debug(f"[{patient_id}] Spacing: {original_spacing}")
    except Exception as e:
        logger.error(f"[{patient_id}] FAILED at spacing extraction: {e}")
        return None
    
    # Step 3: Resample
    try:
        logger.debug(f"[{patient_id}] Resampling volume...")
        resampled = resample_volume(volume, original_spacing, config.target_spacing)
        logger.debug(f"[{patient_id}] Resampled: shape={resampled.shape}")
    except Exception as e:
        logger.error(f"[{patient_id}] FAILED at resampling: {e}")
        logger.debug(traceback.format_exc())
        return None
    
    # Step 4: Windowing
    try:
        logger.debug(f"[{patient_id}] Applying windowing...")
        windowed = apply_windowing(resampled, config.window_center, config.window_width)
        logger.debug(f"[{patient_id}] Windowed: min={windowed.min():.3f}, max={windowed.max():.3f}")
    except Exception as e:
        logger.error(f"[{patient_id}] FAILED at windowing: {e}")
        return None
    
    volume_shape = windowed.shape
    return windowed, volume_shape, original_spacing


def extract_valid_nodules(
    scan,
    patient_id: str,
    config
) -> List[Tuple]:
    """Extract and filter valid nodules from scan."""
    # Step 5: Get nodules
    try:
        nodules = scan.cluster_annotations()
        logger.debug(f"[{patient_id}] Found {len(nodules)} nodule clusters")
    except Exception as e:
        logger.error(f"[{patient_id}] FAILED at cluster_annotations: {e}")
        return []
    
    # Step 6: Extract features for all nodules
    try:
        nodule_features = []
        for idx, annotations in enumerate(nodules):
            features = extract_nodule_features(annotations)
            logger.debug(
                f"[{patient_id}] Nodule {idx}: diameter={features.get('diameter_mm', 'N/A'):.1f}mm, "
                f"annotations={features.get('annotation_count', 0)}"
            )
            nodule_features.append((annotations, features))
    except Exception as e:
        logger.error(f"[{patient_id}] FAILED at feature extraction: {e}")
        logger.debug(traceback.format_exc())
        return []
    
    # Step 7: Filter valid nodules
    def is_valid_nodule(nodule_data):
        annotations, features = nodule_data
        diameter = features.get(RegModelConstants.Features.FEATURE_DIAMETER_MM, 0)
        annot_count = features.get(RegModelConstants.Features.FEATURE_ANNOTATION_COUNT, 0)
        diameter_valid = config.min_nodule_diameter <= diameter <= config.max_nodule_diameter
        return diameter_valid and annot_count > 0
    
    valid_nodules = list(filter(is_valid_nodule, nodule_features))
    logger.debug(f"[{patient_id}] Valid nodules: {len(valid_nodules)}/{len(nodule_features)}")
    
    return valid_nodules


def process_single_slice(
    slice_idx: int,
    nodule_idx: int,
    centroid: Tuple[float, float, float],
    features: Dict,
    windowed: np.ndarray,
    volume_shape: Tuple[int, int, int],
    patient_id: str,
    split: str,
    config,
    directories: Dict[str, Path]
) -> Optional[Dict]:
    """Process single slice: create 2.5D image, compute bbox, save, generate metadata."""
    try:
        # Create 2.5D image
        image_25d = create_25d_sandwich(windowed, slice_idx)
        logger.debug(
            f"[{patient_id}] Slice {slice_idx}: image shape={image_25d.shape}, "
            f"dtype={image_25d.dtype}, range=[{image_25d.min()}, {image_25d.max()}]"
        )
        
        # Validate image
        if image_25d is None or image_25d.size == 0:
            logger.warning(f"[{patient_id}] Slice {slice_idx}: invalid image")
            return None
        
        # Compute bbox
        bbox = compute_nodule_bbox_yolo(
            centroid,
            features[RegModelConstants.Features.FEATURE_DIAMETER_MM],
            volume_shape,
            config.target_spacing,
            config.bbox_padding_factor
        )
        logger.debug(f"[{patient_id}] Slice {slice_idx}: bbox={bbox}")
        
        if bbox is None:
            logger.warning(f"[{patient_id}] Slice {slice_idx}: bbox is None")
            return None
        
        # Validate bbox values
        bbox_valid = all(0 <= v <= 1 for v in bbox)
        if not bbox_valid:
            logger.warning(f"[{patient_id}] Slice {slice_idx}: bbox out of range {bbox}")
            # Don't skip - let's try anyway
        
        # Generate filename and paths
        filename = f"{patient_id}_n{nodule_idx:02d}_z{slice_idx:04d}"
        image_dir = directories[f'{split}_images']
        label_dir = directories[f'{split}_labels']
        image_path = image_dir / f"{filename}.jpg"
        label_path = label_dir / f"{filename}.txt"
        
        logger.debug(f"[{patient_id}] Saving to: {image_path}")
        
        # Atomic save
        save_result = atomic_save_image_and_label(
            image_25d, bbox, config.class_id, image_path, label_path
        )
        
        logger.debug(f"[{patient_id}] Save result: success={save_result.success}")
        
        if not save_result.success:
            logger.warning(
                f"[{patient_id}] Slice {slice_idx}: save failed - {save_result.error_message}"
            )
            return None
        
        # Create metadata entry
        metadata_entry = {
            RegModelConstants.FILE_NAME: filename,
            RegModelConstants.PATIENT_ID: patient_id,
            RegModelConstants.SPLIT_GROUP: split,
            RegModelConstants.NOUDLE_INDEX: nodule_idx,
            RegModelConstants.SLICE_INDEX: slice_idx,
            RegModelConstants.Features.FEATURE_DIAMETER_MM: features[RegModelConstants.Features.FEATURE_DIAMETER_MM],
            f'{RegModelConstants.Features.FEATURE_MALIGNANCY}_score': features[RegModelConstants.Features.FEATURE_MALIGNANCY],
            RegModelConstants.Features.FEATURE_SPICULATION: features[RegModelConstants.Features.FEATURE_SPICULATION],
            RegModelConstants.Features.FEATURE_LOBULATION: features[RegModelConstants.Features.FEATURE_LOBULATION],
            RegModelConstants.Features.FEATURE_SUBTLETY: features[RegModelConstants.Features.FEATURE_SUBTLETY],
            RegModelConstants.Features.FEATURE_SPHERICITY: features[RegModelConstants.Features.FEATURE_SPHERICITY],
            RegModelConstants.Features.FEATURE_MARGIN: features[RegModelConstants.Features.FEATURE_MARGIN],
            RegModelConstants.Features.FEATURE_TEXTURE: features[RegModelConstants.Features.FEATURE_TEXTURE],
            RegModelConstants.Features.FEATURE_CALCIFICATION: features[RegModelConstants.Features.FEATURE_CALCIFICATION],
            RegModelConstants.Features.FEATURE_INTERNAL_STRUCTURE: features[RegModelConstants.Features.FEATURE_INTERNAL_STRUCTURE],
            RegModelConstants.Features.FEATURE_ANNOTATION_COUNT: features[RegModelConstants.Features.FEATURE_ANNOTATION_COUNT],
            RegModelConstants.CENTROID.CENTROID_Z: centroid[0],
            RegModelConstants.CENTROID.CENTROID_Y: centroid[1],
            RegModelConstants.CENTROID.CENTROID_X: centroid[2],
            RegModelConstants.BBOX.BBOX_X: bbox[0],
            RegModelConstants.BBOX.BBOX_Y: bbox[1],
            RegModelConstants.BBOX.BBOX_W: bbox[2],
            RegModelConstants.BBOX.BBOX_H: bbox[3],
            RegModelConstants.IMAGE_PATH: save_result.image_path,
            RegModelConstants.LABEL_PATH: save_result.label_path,
            RegModelConstants.VOLUME.VOLUME_DEPTH: volume_shape[0],
            RegModelConstants.VOLUME.VOLUME_HEIGHT: volume_shape[1],
            RegModelConstants.VOLUME.VOLUME_WIDTH: volume_shape[2]
        }
        logger.debug(f"[{patient_id}] Slice {slice_idx}: SUCCESS")
        return metadata_entry
        
    except Exception as e:
        logger.error(f"[{patient_id}] Slice {slice_idx} error: {e}")
        logger.debug(traceback.format_exc())
        return None


def process_nodule(
    nodule_idx: int,
    annotations,
    features: Dict,
    patient_id: str,
    split: str,
    windowed: np.ndarray,
    volume_shape: Tuple[int, int, int],
    original_spacing: Tuple[float, float, float],
    config,
    directories: Dict[str, Path]
) -> List[Dict]:
    """Process single nodule and generate metadata for its slices."""
    nodule_results = []
    
    # Get centroid
    try:
        centroid = get_nodule_centroid(
            annotations,
            volume_shape,
            original_spacing=original_spacing,
            target_spacing=config.target_spacing
        )
        logger.debug(f"[{patient_id}] Nodule {nodule_idx} centroid: {centroid}")
        
        if centroid is None:
            logger.warning(f"[{patient_id}] Nodule {nodule_idx}: centroid is None, skipping")
            return nodule_results
    except Exception as e:
        logger.error(f"[{patient_id}] Nodule {nodule_idx} centroid error: {e}")
        return nodule_results
    
    # Get slice indices
    try:
        slice_indices = get_nodule_slice_indices(
            annotations,
            volume_shape[0],
            original_spacing=original_spacing,
            target_spacing=config.target_spacing
        )
        logger.debug(f"[{patient_id}] Nodule {nodule_idx} slices: {slice_indices}")
        
        if len(slice_indices) == 0:
            logger.warning(f"[{patient_id}] Nodule {nodule_idx}: no valid slices")
            return nodule_results
    except Exception as e:
        logger.error(f"[{patient_id}] Nodule {nodule_idx} slice indices error: {e}")
        return nodule_results
    
    # Select representative slices
    selected_slices = select_representative_slices(slice_indices, config.slices_per_nodule)
    logger.debug(f"[{patient_id}] Nodule {nodule_idx} selected slices: {selected_slices}")
    
    # Process each slice
    for slice_idx in selected_slices:
        metadata_entry = process_single_slice(
            slice_idx,
            nodule_idx,
            centroid,
            features,
            windowed,
            volume_shape,
            patient_id,
            split,
            config,
            directories
        )
        if metadata_entry is not None:
            nodule_results.append(metadata_entry)
    
    return nodule_results


def process_single_scan(
    scan,
    split: str,
    config,
    directories: Dict[str, Path],
    pl_module
) -> List[Dict]:
    """Process single CT scan (main orchestrator)."""
    metadata_rows = []
    patient_id = scan.patient_id

    logger.debug("=" * 60)
    logger.debug(f"PROCESSING: {patient_id} ({split})")
    logger.debug("=" * 60)
    
    # Steps 1-4: Prepare volume
    volume_result = prepare_scan_volume(scan, patient_id, config)
    if volume_result is None:
        return metadata_rows
    
    windowed, volume_shape, original_spacing = volume_result
    
    # Steps 5-7: Extract and filter nodules
    valid_nodules = extract_valid_nodules(scan, patient_id, config)
    
    if len(valid_nodules) == 0:
        logger.debug(f"[{patient_id}] No valid nodules after filtering")
        return metadata_rows
    
    # Step 8: Process each valid nodule
    for nodule_idx, (annotations, features) in enumerate(valid_nodules):
        nodule_results = process_nodule(
            nodule_idx,
            annotations,
            features,
            patient_id,
            split,
            windowed,
            volume_shape,
            original_spacing,
            config,
            directories
        )
        metadata_rows.extend(nodule_results)
    
    logger.debug("-" * 40)
    logger.debug(f"[{patient_id}] COMPLETE: {len(metadata_rows)} samples generated")
    logger.debug("-" * 40)
    return metadata_rows


def select_representative_slices(slice_indices: List[int], num_slices: int) -> List[int]:
    """Select representative slices from a nodule's slice range."""
    total_slices = len(slice_indices)
    
    if total_slices == 0:
        return []
    
    if total_slices <= num_slices:
        return slice_indices
    
    if num_slices > 1:
        return [
            slice_indices[int(i * (total_slices - 1) / (num_slices - 1))]
            for i in range(num_slices)
        ]
    
    return [slice_indices[total_slices // 2]]

