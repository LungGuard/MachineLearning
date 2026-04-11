"""Shared morphological body/lung segmentation helper.

Used by both SliceQualityGate (lenient thresholds, training) and
DatasetDiagnoser (stricter thresholds, audit) to avoid duplicated
morphological pipeline code.
"""

import cv2
import numpy as np


def compute_lung_body_metrics(
    gray: np.ndarray,
    body_intensity_floor: int = 20,
    lung_intensity_low: int = 10,
    lung_intensity_high: int = 90,
    morph_kernel_size: int = 15,
    min_contour_area: int = 0,
) -> dict:
    """Shared morphological body/lung segmentation.

    Args:
        gray: 2D uint8 image.
        body_intensity_floor: Pixels above this are considered body tissue.
        lung_intensity_low: Lower bound (inclusive) for lung candidate pixels.
        lung_intensity_high: Upper bound (exclusive) for lung candidate pixels.
        morph_kernel_size: Ellipse kernel size for MORPH_CLOSE/OPEN.
        min_contour_area: Minimum contour area to count as a significant lung region.

    Returns:
        dict with keys:
            body_mask    – binary mask (uint8, 0/255)
            body_area    – pixel count of body region
            body_ratio   – body_area / total pixels
            lung_body_ratio – lung area inside body / body_area
            lung_region_count – number of contours > min_contour_area
    """
    total = gray.size

    body_mask = (gray > body_intensity_floor).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel)
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN, kernel)

    body_area = int(np.sum(body_mask > 0))
    body_ratio = body_area / total

    lung_candidate = (
        (gray >= lung_intensity_low) & (gray < lung_intensity_high)
    ).astype(np.uint8) * 255
    lung_in_body = cv2.bitwise_and(lung_candidate, body_mask)

    lung_area = int(np.sum(lung_in_body > 0))
    lung_body_ratio = lung_area / body_area if body_area > 0 else 0.0

    contours, _ = cv2.findContours(lung_in_body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant = [c for c in contours if cv2.contourArea(c) > min_contour_area]

    return {
        "body_mask": body_mask,
        "body_area": body_area,
        "body_ratio": body_ratio,
        "lung_body_ratio": lung_body_ratio,
        "lung_region_count": len(significant),
    }
