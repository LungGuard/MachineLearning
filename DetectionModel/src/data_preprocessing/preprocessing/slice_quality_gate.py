"""Slice Quality Gate — Pre-save validation and CLAHE enhancement.

Validates and enhances individual 2D slices before they enter
either the data-prep or inference pipeline.

Two jobs:
  1. Reject slices that won't be useful (apex/base, too bright, no lung)
  2. Apply CLAHE to spread contrast → eliminates LOW_CONTRAST at source
"""

import logging
from dataclasses import dataclass
from typing import Tuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SliceQualityConfig:
    """Pre-save validation thresholds.

    Aligned with the diagnoser thresholds so that slices passing
    this gate will also pass the diagnoser.
    """
    body_intensity_floor: int = 20
    lung_intensity_low: int = 10
    lung_intensity_high: int = 90
    morph_kernel_size: int = 15
    min_lung_body_ratio: float = 0.12
    min_contrast_range: int = 80
    max_mean_brightness: float = 180.0
    min_dark_ratio: float = 0.20
    dark_threshold: int = 50
    clahe_clip_limit: float = 2.5
    clahe_grid_size: Tuple[int, int] = (8, 8)


class SliceQualityGate:
    """Validates and enhances a 2D slice (single- or multi-channel)."""

    def __init__(self, config: SliceQualityConfig = None):
        self.config = config or SliceQualityConfig()

    def validate_and_enhance(self, image: np.ndarray,
                              patient_id: str = "",
                              context: str = "") -> Tuple[Optional[np.ndarray], bool, str]:
        """Apply CLAHE, then validate.

        Returns:
            (enhanced_image_or_None, passed, reason)
        """
        enhanced = self._apply_clahe(image)
        passed, reason = self._check_quality(enhanced)

        logger.debug(
            f"[{patient_id}] {context} rejected: {reason}"
        ) if not passed else None

        result_image = enhanced if passed else None
        return result_image, passed, reason

    # ── Quality Checks ────────────────────────

    def _check_quality(self, image: np.ndarray) -> Tuple[bool, str]:
        """All quality checks in one pass."""
        c = self.config
        check_slice = image[:, :, image.shape[2] // 2] if len(image.shape) == 3 else image
        gray = self._to_uint8(check_slice)
        total = gray.size

        mean_val = float(gray.mean())
        max_val = int(gray.max())
        min_val = int(gray.min())
        contrast_range = max_val - min_val
        dark_ratio = int(np.sum(gray < c.dark_threshold)) / total
        lung_body_ratio = self._compute_lung_ratio(gray)

        checks = [
            (contrast_range < c.min_contrast_range, f"LOW_CONTRAST (range={contrast_range})"),
            (dark_ratio < c.min_dark_ratio and mean_val > 100, f"NO_BG (dark_ratio={dark_ratio:.3f})"),
            (mean_val > c.max_mean_brightness, f"TOO_BRIGHT (mean={mean_val:.1f})"),
            (lung_body_ratio < c.min_lung_body_ratio, f"INSUFFICIENT_LUNG (ratio={lung_body_ratio:.3f})"),
        ]

        failures = list(filter(lambda chk: chk[0], checks))
        passed = len(failures) == 0
        reason = failures[0][1] if failures else "OK"
        return passed, reason

    def _compute_lung_ratio(self, gray: np.ndarray) -> float:
        """Lung-to-body area ratio (same algorithm as diagnoser)."""
        c = self.config
        body_mask = (gray > c.body_intensity_floor).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (c.morph_kernel_size, c.morph_kernel_size)
        )
        body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel)
        body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN, kernel)
        body_area = int(np.sum(body_mask > 0))

        lung_candidate = (
            (gray >= c.lung_intensity_low) & (gray < c.lung_intensity_high)
        ).astype(np.uint8) * 255
        lung_in_body = cv2.bitwise_and(lung_candidate, body_mask)
        lung_area = int(np.sum(lung_in_body > 0))

        ratio = lung_area / body_area if body_area > 0 else 0.0
        return ratio

    # ── CLAHE Enhancement ─────────────────────

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """CLAHE on single- or multi-channel images."""
        c = self.config
        clahe = cv2.createCLAHE(clipLimit=c.clahe_clip_limit, tileGridSize=c.clahe_grid_size)
        result = (
            np.stack(
                list(map(lambda ch: clahe.apply(self._to_uint8(image[:, :, ch])), range(image.shape[2]))),
                axis=2
            )
            if len(image.shape) == 3
            else clahe.apply(self._to_uint8(image))
        )
        return result

    @staticmethod
    def _to_uint8(arr: np.ndarray) -> np.ndarray:
        is_float_01 = arr.dtype in (np.float32, np.float64) and arr.max() <= 1.0
        converted = (
            (arr * 255).clip(0, 255).astype(np.uint8)
            if is_float_01
            else arr.clip(0, 255).astype(np.uint8)
        )
        return converted