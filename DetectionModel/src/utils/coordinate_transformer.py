"""
Coordinate Transformer Module
Utilities for transforming coordinates between volume spaces.
"""

from typing import Tuple


class CoordinateTransformer:
    """Utilities for transforming coordinates between volume spaces."""
    
    @staticmethod
    def transform_coordinates_to_resampled(
        original_coords: Tuple[float, float, float],
        original_spacing: Tuple[float, float, float],
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> Tuple[float, float, float]:
        """Transform coordinates from original volume space to resampled volume space."""
        scale_factors = tuple(
            orig / tgt for orig, tgt in zip(original_spacing, target_spacing)
        )
        
        transformed = tuple(
            coord * scale for coord, scale in zip(original_coords, scale_factors)
        )
        
        return transformed
    
    @staticmethod
    def transform_slice_to_resampled_space(orig_idx: int, z_scale: float) -> int:
        """Transform a slice index from original space to resampled space."""
        return int(round(orig_idx * z_scale))
    
    @staticmethod
    def is_slice_within_volume(slice_idx: int, volume_depth: int) -> bool:
        """Check if slice index is within valid volume bounds."""
        return 0 <= slice_idx < volume_depth
