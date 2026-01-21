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


def atomic_save_image_and_label(
    image: np.ndarray,
    yolo_bbox: Tuple[float, float, float, float],
    class_id: int,
    image_path: Path,
    label_path: Path
) -> AtomicSaveResult:
    """Save image and label atomically (label only if image succeeds)."""
    result = AtomicSaveResult(success=False)
    
    # Attempt image save
    try:
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        save_success = cv2.imwrite(str(image_path), image_bgr)
        
        image_saved = save_success and image_path.exists()
    except Exception as e:
        logger.error(f"Image save failed: {e}")
        image_saved = False
    
    # Save label ONLY if image save succeeded
    try:
        label_content = (
            f"{class_id} "
            f"{yolo_bbox[0]:.6f} "
            f"{yolo_bbox[1]:.6f} "
            f"{yolo_bbox[2]:.6f} "
            f"{yolo_bbox[3]:.6f}\n"
        )
        
        # Write label file only if image was saved
        write_label = image_saved
        
        label_saved = False
        if write_label:
            with open(label_path, 'w') as f:
                f.write(label_content)
            label_saved = True
            
    except Exception as e:
        logger.error(f"Label save failed: {e}")
        label_saved = False
        # Rollback: delete image if label failed
        if image_saved and image_path.exists():
            image_path.unlink()
        image_saved = False
    
    result = AtomicSaveResult(
        success=image_saved and label_saved,
        image_path=str(image_path) if image_saved else None,
        label_path=str(label_path) if label_saved else None,
        error_message=None if (image_saved and label_saved) else "Save operation failed"
    )
    
    return result
