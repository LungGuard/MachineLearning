"""
Volume Preprocessor Module
Utilities for preprocessing CT volume data.
"""

import numpy as np
from scipy.ndimage import zoom
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class VolumePreprocessor:
    """Utilities for preprocessing CT volume data."""
    
    @staticmethod
    def resample_volume(
        volume: np.ndarray,
        original_spacing: Tuple[float, float, float],
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> np.ndarray:
        """
        Resample a 3D CT volume to isotropic voxel spacing.
        
        Medical CT scans have non-uniform voxel spacing (e.g., 0.7mm x 0.7mm x 2.5mm).
        This function resamples to uniform spacing for consistent analysis.
        
        Returns a Resampled 3D volume with target spacing
        """
        zoom_factors = tuple(
            orig / target 
            for orig, target in zip(original_spacing, target_spacing)
        )
        
        resampled = zoom(volume, zoom_factors, order=1, mode='nearest')
        
        logger.debug(
            f"Resampled volume from {volume.shape} to {resampled.shape} "
            f"(spacing: {original_spacing} -> {target_spacing})"
        )
        
        return resampled
    
    @staticmethod
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

        Returns a Windowed volume, optionally normalized to [0, 1]
        
        Notes
        -----
        Lung HU ranges:
            - Air: -1000 HU
            - Lung tissue: -500 to -900 HU
            - Soft tissue: -100 to 100 HU
            - Bone: 400+ HU
        """
        window_lower_bound = center - (width / 2.0)
        window_upper_bound = center + (width / 2.0)
        
        windowed = np.clip(volume, window_lower_bound, window_upper_bound)
        
        normalized = (
            (windowed - window_lower_bound) / width 
            if normalize 
            else windowed
        )
        
        return normalized.astype(np.float32)
    
    @staticmethod
    def create_25d_sandwich(
        volume: np.ndarray,
        z_index: int,
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
        
        z_prev = max(0, z_index - 1)
        z_curr = z_index
        z_next = min(depth - 1, z_index + 1)
        
        slice_prev = volume[z_prev]
        slice_curr = volume[z_curr]
        slice_next = volume[z_next]
        
        sandwich = np.stack([slice_prev, slice_curr, slice_next], axis=-1)
        rgb_image = (sandwich * 255.0).astype(np.uint8)
        
        return rgb_image
