"""Atomic file I/O for image/label pairs."""

import logging
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass
import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class AtomicSaveResult:
    """Save operation result."""
    success: bool
    image_path: Optional[str] = None
    label_path: Optional[str] = None
    error_message: Optional[str] = None


def save_image(image: np.ndarray, image_path: Path) -> bool:
    """
    Save image to disk.
    Returns Saving result
    """
    try:
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        save_success = cv2.imwrite(str(image_path), image_bgr)
        
        return save_success and image_path.exists()
    except Exception as e:
        logger.error(f"Image save failed: {e}")
        return False


def save_label(
    class_id: int,
    yolo_bbox: Tuple[float, float, float, float],
    label_path: Path
) -> bool:
    """
    Save YOLO format label to disk.
    Returns Saving result
    """
    try:
        label_content = (
            f"{class_id} "
            f"{yolo_bbox[0]:.6f} "
            f"{yolo_bbox[1]:.6f} "
            f"{yolo_bbox[2]:.6f} "
            f"{yolo_bbox[3]:.6f}\n"
        )
        
        with open(label_path, 'w') as f:
            f.write(label_content)
        
        return label_path.exists()
    except Exception as e:
        logger.error(f"Label save failed: {e}")
        return False


def atomic_save_image_and_label(
    image: np.ndarray,
    yolo_bbox: Tuple[float, float, float, float],
    class_id: int,
    image_path: Path,
    label_path: Path
) -> AtomicSaveResult:
    """
    Save image and label atomically (label only if image succeeds).
    If label save fails, the image is rolled back (deleted).
    
    Args:
        image: Image array in RGB format
        yolo_bbox: YOLO format bounding box (x_center, y_center, width, height)
        class_id: Class ID for YOLO format
        image_path: Path where to save the image
        label_path: Path where to save the label
        
    Returns:
        AtomicSaveResult with success status and file paths
    """
    image_saved = save_image(image, image_path)
    
    if not image_saved:
        return AtomicSaveResult(
            success=False,
            error_message="Image save failed"
        )
    
    label_saved = save_label(class_id, yolo_bbox, label_path)
    
    if not label_saved:
        if image_path.exists():
            try:
                image_path.unlink()
                logger.debug(f"Rolled back image deletion: {image_path}")
            except Exception as e:
                logger.error(f"Failed to rollback image: {e}")
        
        return AtomicSaveResult(
            success=False,
            error_message="Label save failed - image rolled back"
        )
    
    return AtomicSaveResult(
        success=True,
        image_path=str(image_path),
        label_path=str(label_path),
        error_message=None
    )
