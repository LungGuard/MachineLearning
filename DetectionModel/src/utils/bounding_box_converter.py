"""
Bounding Box Converter Module
Utilities for converting bounding boxes to YOLO format.
"""

import numpy as np
from typing import Tuple, Optional


class BoundingBoxConverter:
    """Utilities for converting bounding boxes to YOLO format."""
    
    @staticmethod
    def compute_diameter(bbox) -> float:
        """Compute nodule diameter from bounding box."""
        x_extent = bbox[2][1] - bbox[2][0]
        y_extent = bbox[1][1] - bbox[1][0]
        diameter = np.sqrt(x_extent**2 + y_extent**2)
        return float(diameter)
    
    @staticmethod
    def convert_to_yolo_format(
        volume_shape: Tuple[int, int, int],
        spacing: Tuple[float, float, float],
        nodule_centroid: Tuple[float, float, float],
        padding_factor: float,
        nodule_diameter: float
    ) -> Tuple[float, float, float, float]:
        """Convert nodule parameters to YOLO normalized format."""
        depth, height, width = volume_shape
        z_spacing, y_spacing, x_spacing = spacing
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
        w_norm = np.clip(w_norm, 0.001, 1.0)
        h_norm = np.clip(h_norm, 0.001, 1.0)

        return x_norm, y_norm, w_norm, h_norm
    
    @staticmethod
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
        x_norm, y_norm, w_norm, h_norm = BoundingBoxConverter.convert_to_yolo_format(
            volume_shape, spacing, nodule_centroid, padding_factor, nodule_diameter
        )
        
        is_valid_bbox_bounds = (
            (x_norm - w_norm/2 >= 0) and 
            (x_norm + w_norm/2 <= 1) and
            (y_norm - h_norm/2 >= 0) and 
            (y_norm + h_norm/2 <= 1)
        )
        
        result = (
            (float(x_norm), float(y_norm), float(w_norm), float(h_norm))
            if is_valid_bbox_bounds
            else (float(x_norm), float(y_norm), 
                  float(min(w_norm, 2*min(x_norm, 1-x_norm))), 
                  float(min(h_norm, 2*min(y_norm, 1-y_norm))))
        )
        
        return result
