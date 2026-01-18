"""
LungGuard Data Preparation - Utils Module
=========================================
Pure functions for medical image manipulation.
NO file I/O operations - this is a pure computational library.

Author: LungGuard ML Team
License: Proprietary
"""

import numpy as np
from scipy.ndimage import zoom
from typing import Tuple, Optional
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
    
    Parameters
    ----------
    volume : np.ndarray
        3D numpy array of shape (Z, H, W), should be pre-windowed and normalized
    z_index : int
        Central slice index
    pad_mode : str
        Padding mode for edge slices ('edge', 'constant', 'reflect')
    
    Returns
    -------
    np.ndarray
        2.5D RGB image of shape (H, W, 3) with values in [0, 255] uint8
    
    Notes
    -----
    Edge handling:
        - z_index=0: uses [0, 0, 1] slices
        - z_index=max: uses [max-1, max, max] slices
    
    T(N) = O(H * W) where H, W are slice dimensions
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
    
    Parameters
    ----------
    nodule_centroid : Tuple[float, float, float]
        Nodule center coordinates (z, y, x) in voxel space
    nodule_diameter : float
        Nodule diameter in mm
    volume_shape : Tuple[int, int, int]
        Volume dimensions (depth, height, width)
    spacing : Tuple[float, float, float]
        Voxel spacing (z, y, x) in mm
    padding_factor : float
        Multiplicative padding for bounding box (1.5 = 50% larger)
    
    Returns
    -------
    Optional[Tuple[float, float, float, float]]
        YOLO bbox (x_center, y_center, width, height) normalized, or None if invalid

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
    
    Parameters
    ----------
    annotations : list
        List of pylidc Annotation objects for a single nodule
    fallback_diameter : float
        Default diameter in mm when computation fails
    
    Returns
    -------
    dict
        Aggregated nodule features:
        - diameter_mm: Average diameter
        - malignancy: Average malignancy score (1-5)
        - spiculation: Average spiculation score (1-5)
        - lobulation: Average lobulation score (1-5)
        - subtlety: Average subtlety score (1-5)
        - sphericity: Average sphericity score (1-5)
        - margin: Average margin definition (1-5)
        - texture: Average texture score (1-5)
        - calcification: Average calcification score (1-6)
        - internal_structure: Average internal structure (1-4)
        - annotation_count: Number of annotations
    
    Notes
    -----
    Malignancy interpretation:
        1 = Highly Unlikely
        2 = Moderately Unlikely  
        3 = Indeterminate
        4 = Moderately Suspicious
        5 = Highly Suspicious
    
    T(N) = O(K) where K is number of annotations (typically 1-4)
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
    annotations: list,
    volume_depth: int
) -> list:
    """
    Get all Z-slice indices that contain the nodule across all annotations.
    
    Parameters
    ----------
    annotations : list
        List of pylidc Annotation objects
    volume_depth : int
        Total number of slices in the volume
    
    Returns
    -------
    list
        Sorted list of unique Z-indices containing the nodule
    
    Notes
    -----
    T(N) = O(K * S) where K is annotations, S is slices per annotation
    """
    slice_indices = set()
    
    for ann in annotations:
        try:
            bbox = ann.bbox()
            z_min, z_max = int(bbox[0][0]), int(bbox[0][1])
            valid_indices = [
                z for z in range(z_min, z_max + 1)
                if 0 <= z < volume_depth
            ]
            slice_indices.update(valid_indices)
        except Exception:
            pass
    
    result = sorted(list(slice_indices))
    
    return result


def get_nodule_centroid(
    annotations: list,
    volume_shape: Tuple[int, int, int]
) -> Optional[Tuple[float, float, float]]:
    """
    Compute the average centroid of a nodule from all annotations.
    
    Parameters
    ----------
    annotations : list
        List of pylidc Annotation objects
    volume_shape : Tuple[int, int, int]
        Volume dimensions (depth, height, width)
    
    Returns
    -------
    Optional[Tuple[float, float, float]]
        Average centroid (z, y, x) in voxel coordinates, or None if invalid
    
    Notes
    -----
    T(N) = O(K) where K is number of annotations
    """
    depth, height, width = volume_shape
    
    centroids = []
    for ann in annotations:
        try:
            centroid = ann.centroid
            # Validate centroid is within bounds
            is_valid = (
                0 <= centroid[0] < depth and
                0 <= centroid[1] < height and
                0 <= centroid[2] < width
            )
            valid_centroids = [centroid] if is_valid else []
            centroids.extend(valid_centroids)
        except Exception:
            pass
    
    result = (
        tuple(np.mean(centroids, axis=0).tolist())
        if centroids
        else None
    )
    
    return result