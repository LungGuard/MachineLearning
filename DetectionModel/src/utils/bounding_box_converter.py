"""
Bounding Box Converter Module
Utilities for converting bounding boxes to YOLO format.
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BoundingBoxConverter:
    """Utilities for converting bounding boxes to YOLO format."""
    
    @staticmethod
    def compute_diameter(bbox) -> float:
        """Compute nodule diameter from bounding box."""
        # bbox is a tuple of slice objects: (slice(z1,z2), slice(y1,y2), slice(x1,x2))
        x_extent = bbox[2].stop - bbox[2].start
        y_extent = bbox[1].stop - bbox[1].start
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

    @staticmethod
    def adjust_bbox_for_resize(
        original_bbox: Tuple[float, float, float, float],
        original_size: Tuple[int, int],
        target_size: Tuple[int, int],
        preserve_aspect_ratio: bool = True
    ) -> Tuple[float, float, float, float]:
        """
        Adjust YOLO bbox coordinates when image is resized.

        When an image is resized (especially with aspect ratio preservation and padding),
        the bounding box coordinates must be transformed to match the new image space.

        Args:
            original_bbox: (x_center, y_center, width, height) in normalized [0, 1] coordinates
            original_size: (height, width) of original image
            target_size: (height, width) of target image
            preserve_aspect_ratio: Whether aspect ratio was preserved (with padding)

        Returns:
            Adjusted bbox (x_center, y_center, width, height) in normalized coordinates
        """
        x_c, y_c, w, h = original_bbox
        orig_h, orig_w = original_size
        target_h, target_w = target_size

        if not preserve_aspect_ratio:
            # Simple resize: coordinates remain the same in normalized space
            return original_bbox

        # Calculate scale and offsets for aspect-preserved resize
        scale = min(target_h / orig_h, target_w / orig_w)
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)

        y_offset = (target_h - new_h) / 2.0
        x_offset = (target_w - new_w) / 2.0

        # Convert from normalized original to pixel original
        x_c_px = x_c * orig_w
        y_c_px = y_c * orig_h
        w_px = w * orig_w
        h_px = h * orig_h

        # Apply scale and offset
        x_c_px_scaled = x_c_px * scale + x_offset
        y_c_px_scaled = y_c_px * scale + y_offset
        w_px_scaled = w_px * scale
        h_px_scaled = h_px * scale

        # Convert back to normalized for target size
        x_c_new = x_c_px_scaled / target_w
        y_c_new = y_c_px_scaled / target_h
        w_new = w_px_scaled / target_w
        h_new = h_px_scaled / target_h

        # Clamp to valid range
        x_c_new = np.clip(x_c_new, 0.0, 1.0)
        y_c_new = np.clip(y_c_new, 0.0, 1.0)
        w_new = np.clip(w_new, 0.001, 1.0)
        h_new = np.clip(h_new, 0.001, 1.0)

        logger.debug(
            f"Bbox adjusted for resize: ({x_c:.4f}, {y_c:.4f}, {w:.4f}, {h:.4f}) -> "
            f"({x_c_new:.4f}, {y_c_new:.4f}, {w_new:.4f}, {h_new:.4f})"
        )

        return (float(x_c_new), float(y_c_new), float(w_new), float(h_new))
