"""
LungGuard Data Preparation - Utils Module
=========================================
Pure functions for medical image manipulation.
NO file I/O operations - this is a pure computational library.

"""

import numpy as np
from scipy.ndimage import zoom
from typing import Tuple, Optional,List
import logging
from constants.detection.dataset_constants import RegModelConstants
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def resample_volume(
    volume: np.ndarray,
    original_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> np.ndarray:
    """
    Resample a 3D CT volume to isotropic voxel spacing.
    
    Medical CT scans have non-uniform voxel spacing (e.g., 0.7mm x 0.7mm x 2.5mm).
    This function resamples to uniform spacing for consistent analysis.
    
    Parameters
    ----------
    volume : np.ndarray
        3D numpy array of shape (Z, Y, X) containing Hounsfield Units
    original_spacing : Tuple[float, float, float]
        Original voxel spacing in mm (z_spacing, y_spacing, x_spacing)
    target_spacing : Tuple[float, float, float]
        Target voxel spacing in mm, default (1.0, 1.0, 1.0) for isotropic
    
    Returns
    -------
    np.ndarray
        Resampled 3D volume with target spacing
    
    Notes
    -----
    Uses spline interpolation (order=1 for speed, order=3 for quality).
    T(N) = O(N * M) where N is original voxels, M is resampled voxels
    """
    # Calculate zoom factors for each dimension
    zoom_factors = tuple(
        orig / target 
        for orig, target in zip(original_spacing, target_spacing)
    )
    
    # Apply resampling with linear interpolation (order=1)
    # order=1 is faster and sufficient for CT data
    resampled = zoom(volume, zoom_factors, order=1, mode='nearest')
    
    logger.debug(
        f"Resampled volume from {volume.shape} to {resampled.shape} "
        f"(spacing: {original_spacing} -> {target_spacing})"
    )
    
    return resampled


def apply_windowing(
    volume: np.ndarray,
    center: float = -600.0,
    width: float = 1500.0,
    normalize: bool = True
) -> np.ndarray:
    """
    Apply CT windowing (window level/width) for lung visualization.
    
    CT values are in Hounsfield Units (HU). Different tissues have different HU ranges.
    Windowing maps a specific HU range to display values for optimal visualization.
    
    Standard lung window: center=-600, width=1500 (range: -1350 to 150 HU)
    
    Parameters
    ----------
    volume : np.ndarray
        3D numpy array containing Hounsfield Units
    center : float
        Window center (level) in HU, default -600 for lung
    width : float
        Window width in HU, default 1500 for lung
    normalize : bool
        If True, normalize output to [0, 1] range
    
    Returns
    -------
    np.ndarray
        Windowed volume, optionally normalized to [0, 1]
    
    Notes
    -----
    Lung HU ranges:
        - Air: -1000 HU
        - Lung tissue: -500 to -900 HU
        - Soft tissue: -100 to 100 HU
        - Bone: 400+ HU
    
    T(N) = O(N) where N is total voxels
    """
    # Calculate window bounds
    lower_bound = center - (width / 2.0)
    upper_bound = center + (width / 2.0)
    
    # Apply windowing using clip
    windowed = np.clip(volume, lower_bound, upper_bound)
    
    # Normalize to [0, 1] range using vectorized operations
    normalized = (
        (windowed - lower_bound) / width 
        if normalize 
        else windowed
    )
    
    return normalized.astype(np.float32)


def create_25d_sandwich(
    volume: np.ndarray,
    z_index: int,
    pad_mode: str = 'edge'
) -> np.ndarray:
    """
    Create a 2.5D RGB image from adjacent CT slices.
    
    2.5D representation uses 3 adjacent slices as RGB channels,
    capturing spatial context while maintaining 2D compatibility with YOLO.
    
    The "sandwich" approach:
        - Red channel: slice at z_index - 1 (previous)
        - Green channel: slice at z_index (current/target)
        - Blue channel: slice at z_index + 1 (next)
    """
    depth, height, width = volume.shape
    
    # Calculate slice indices with boundary handling
    # Using max/min avoids conditional logic
    z_prev = max(0, z_index - 1)
    z_curr = z_index
    z_next = min(depth - 1, z_index + 1)
    
    # Extract slices for RGB channels
    slice_prev = volume[z_prev]
    slice_curr = volume[z_curr]
    slice_next = volume[z_next]
    
    # Stack into RGB image (H, W, 3)
    sandwich = np.stack([slice_prev, slice_curr, slice_next], axis=-1)
    
    # Convert to uint8 [0, 255] for image saving
    # Assumes input is normalized to [0, 1]
    rgb_image = (sandwich * 255.0).astype(np.uint8)
    
    return rgb_image


def compute_nodule_bbox_yolo(
    nodule_centroid: Tuple[float, float, float],
    nodule_diameter: float,
    volume_shape: Tuple[int, int, int],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    padding_factor: float = 1.5
) -> Optional[Tuple[float, float, float, float]]:
    """
    Compute YOLO-format bounding box from nodule centroid and diameter.
    
    YOLO format: (x_center, y_center, width, height) normalized to [0, 1]
    
    """
    depth, height, width = volume_shape
    z_spacing, y_spacing, x_spacing = spacing
    
    # Extract centroid coordinates
    z_center, y_center, x_center = nodule_centroid
    
    # Convert diameter from mm to voxels (using x,y spacing for 2D bbox)
    diameter_voxels_x = (nodule_diameter / x_spacing) * padding_factor
    diameter_voxels_y = (nodule_diameter / y_spacing) * padding_factor
    
    # Normalize to [0, 1] for YOLO format
    x_norm = x_center / width
    y_norm = y_center / height
    w_norm = diameter_voxels_x / width
    h_norm = diameter_voxels_y / height
    
    # Clamp values to valid range [0, 1]
    x_norm = np.clip(x_norm, 0.0, 1.0)
    y_norm = np.clip(y_norm, 0.0, 1.0)
    w_norm = np.clip(w_norm, 0.001, 1.0)  # Minimum size
    h_norm = np.clip(h_norm, 0.001, 1.0)
    
    # Validate bbox is within bounds
    is_valid = (
        (x_norm - w_norm/2 >= 0) and 
        (x_norm + w_norm/2 <= 1) and
        (y_norm - h_norm/2 >= 0) and 
        (y_norm + h_norm/2 <= 1)
    )
    
    result = (
        (float(x_norm), float(y_norm), float(w_norm), float(h_norm))
        if is_valid
        else (float(x_norm), float(y_norm), float(min(w_norm, 2*min(x_norm, 1-x_norm))), 
              float(min(h_norm, 2*min(y_norm, 1-y_norm))))
    )
    
    return result


def extract_nodule_features(
    annotations: list,
    fallback_diameter: float = 10.0
) -> dict:
    """
    Extract and aggregate features from multiple radiologist annotations.
    
    LIDC-IDRI nodules have 1-4 independent radiologist annotations.
    This function computes consensus features using averaging.
    """
    # Default feature dictionary
    default_features = {
        RegModelConstants.Features.FEATURE_DIAMETER_MM : fallback_diameter,
        RegModelConstants.Features.FEATURE_MALIGNANCY : 3.0,  # Indeterminate
        RegModelConstants.Features.FEATURE_SPICULATION : 1.0,
        RegModelConstants.Features.FEATURE_LOBULATION : 1.0,
        RegModelConstants.Features.FEATURE_SUBTLETY : 3.0,
        RegModelConstants.Features.FEATURE_SPHERICITY : 3.0,
        RegModelConstants.Features.FEATURE_MARGIN : 3.0,
        RegModelConstants.Features.FEATURE_TEXTURE : 3.0,
        RegModelConstants.Features.FEATURE_CALCIFICATION : 1.0,
        RegModelConstants.Features.FEATURE_INTERNAL_STRUCTURE : 1.0,
        RegModelConstants.Features.FEATURE_ANNOTATION_COUNT: 0 
    }
    
    # Handle empty annotations
    has_annotations = len(annotations) > 0
    
    # Extract features using list comprehensions (avoiding explicit loops with break/continue)
    malignancy_scores = [ann.malignancy for ann in annotations] if has_annotations else []
    spiculation_scores = [ann.spiculation for ann in annotations] if has_annotations else []
    lobulation_scores = [ann.lobulation for ann in annotations] if has_annotations else []
    subtlety_scores = [ann.subtlety for ann in annotations] if has_annotations else []
    sphericity_scores = [ann.sphericity for ann in annotations] if has_annotations else []
    margin_scores = [ann.margin for ann in annotations] if has_annotations else []
    texture_scores = [ann.texture for ann in annotations] if has_annotations else []
    calcification_scores = [ann.calcification for ann in annotations] if has_annotations else []
    internal_structure_scores = [ann.internalStructure for ann in annotations] if has_annotations else []
    
    # Compute diameters using bounding boxes
    diameters = []
    for ann in annotations:
        try:
            bbox = ann.bbox()
            # bbox returns ((z_min, z_max), (y_min, y_max), (x_min, x_max))
            x_extent = bbox[2][1] - bbox[2][0]
            y_extent = bbox[1][1] - bbox[1][0]
            diameter = np.sqrt(x_extent**2 + y_extent**2)
            diameters.append(float(diameter))
        except Exception:
            pass
    
    # Compute averages using numpy for efficiency
    features = {
        RegModelConstants.Features.FEATURE_DIAMETER_MM : float(np.mean(diameters)) if diameters else fallback_diameter,
        RegModelConstants.Features.FEATURE_MALIGNANCY : float(np.mean(malignancy_scores)) if malignancy_scores else 3.0,
        RegModelConstants.Features.FEATURE_SPICULATION : float(np.mean(spiculation_scores)) if spiculation_scores else 1.0,
        RegModelConstants.Features.FEATURE_LOBULATION : float(np.mean(lobulation_scores)) if lobulation_scores else 1.0,
        RegModelConstants.Features.FEATURE_SUBTLETY : float(np.mean(subtlety_scores)) if subtlety_scores else 3.0,
        RegModelConstants.Features.FEATURE_SPHERICITY : float(np.mean(sphericity_scores)) if sphericity_scores else 3.0,
        RegModelConstants.Features.FEATURE_MARGIN : float(np.mean(margin_scores)) if margin_scores else 3.0,
        RegModelConstants.Features.FEATURE_TEXTURE : float(np.mean(texture_scores)) if texture_scores else 3.0,
        RegModelConstants.Features.FEATURE_CALCIFICATION: float(np.mean(calcification_scores)) if calcification_scores else 1.0,
        RegModelConstants.Features.FEATURE_INTERNAL_STRUCTURE: float(np.mean(internal_structure_scores)) if internal_structure_scores else 1.0,
        RegModelConstants.Features.FEATURE_ANNOTATION_COUNT: len(annotations)
    }
    
    result = features if has_annotations else default_features
    
    return result


def get_nodule_slice_indices(
    annotations: List,
    volume_depth: int,
    original_spacing: Tuple[float, float, float] = None,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> List[int]:
    """
    Get valid slice indices for a nodule, transformed to resampled space.
    """
    slice_set = set()
    
    # Calculate Z-axis scale factor
    z_scale = (
        original_spacing[0] / target_spacing[0]
        if original_spacing is not None
        else 1.0
    )
    
    for ann in annotations:
        try:
            # Get slice indices from annotation (in original space)
            contour_slice_indices = ann.contour_slice_indices
            
            # Transform each slice index to resampled space
            for orig_idx in contour_slice_indices:
                transformed_idx = int(round(orig_idx * z_scale))
                is_valid = 0 <= transformed_idx < volume_depth
                slice_set.add(transformed_idx) if is_valid else None
        except Exception:
            pass  # Skip failed annotations
    
    return sorted(list(slice_set))

def transform_coordinates_to_resampled(
    original_coords: Tuple[float, float, float],
    original_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Tuple[float, float, float]:
    """
    Transform coordinates from original volume space to resampled volume space.
    """
    scale_factors = tuple(
        orig / tgt for orig, tgt in zip(original_spacing, target_spacing)
    )
    
    transformed = tuple(
        coord * scale for coord, scale in zip(original_coords, scale_factors)
    )
    
    return transformed


def get_nodule_centroid(
    annotations,
    volume_shape: Tuple[int, int, int],
    original_spacing: Tuple[float, float, float] = None,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Optional[Tuple[float, float, float]]:
    """
    Calculate nodule centroid from annotations, transformed to resampled space.
    """
    result = None
    
    centroids = []
    for ann in annotations:
        try:
            centroid = ann.centroid  # Returns (z, y, x) in original space
            is_valid = centroid is not None and len(centroid) == 3
            centroids.append(centroid) if is_valid else None
        except Exception:
            pass  # Skip failed annotations
    
    if len(centroids) > 0:
        # Average centroid in original space
        avg_centroid = tuple(
            sum(c[i] for c in centroids) / len(centroids)
            for i in range(3)
        )
        
        # Transform to resampled space if spacing provided
        transformed_centroid = (
            transform_coordinates_to_resampled(avg_centroid, original_spacing, target_spacing)
            if original_spacing is not None
            else avg_centroid
        )
        
        # Validate against resampled volume bounds
        z, y, x = transformed_centroid
        is_within_bounds = (
            0 <= z < volume_shape[0] and
            0 <= y < volume_shape[1] and
            0 <= x < volume_shape[2]
        )
        
        result = transformed_centroid if is_within_bounds else None
    
    return result