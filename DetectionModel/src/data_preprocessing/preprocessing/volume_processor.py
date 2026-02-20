"""Volume Preprocessing — MONAI-powered CT volume pipeline.

Handles the heavy lifting between raw DICOM data and model-ready volumes:
  1. NaN cleaning
  2. Padding/offset detection and correction
  3. Resampling to isotropic spacing
  4. Lung windowing with intensity normalization
"""

import logging
import traceback
from typing import Tuple, Optional

import numpy as np
import torch
from monai.transforms import ScaleIntensityRange, Zoom

from DetectionModel.constants.constants.preprocessing import PreProcessingConstants, IntensityRange
from DetectionModel.constants.enums.hu_values import HUValues

logger = logging.getLogger(__name__)



class VolumePreprocessingPipeline:
    """Cleans, resamples, and windows a raw CT volume.

    Source-agnostic — works on any (D, H, W) numpy array + spacing tuple.
    """

    def __init__(self, config):
        """
        Args:
            config: must have .target_spacing and optionally
                    .window_center, .window_width
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def preprocess(self, raw_volume: np.ndarray,
                   original_spacing: Tuple[float, float, float],
                   patient_id: str = "") -> Optional[Tuple[np.ndarray, Tuple, Tuple]]:
        """Full preprocessing pipeline.

        Returns:
            (final_volume, volume_shape, original_spacing) or None on failure.
        """
        try:
            return self._perform_full_preprocess(
                raw_volume, patient_id, original_spacing
            )
        except Exception as e:
            self.logger.error(f"[{patient_id}] MONAI Processing FAILED: {e}")
            self.logger.debug(traceback.format_exc())
            return None

    def _perform_full_preprocess(self, raw_volume, patient_id, original_spacing):
        cleaned = self._clean_with_offset_detection(
            raw_volume.astype(np.float32), patient_id
        )

        volume_tensor = torch.from_numpy(cleaned).float().unsqueeze(0)
        volume_tensor = torch.clamp(
            volume_tensor, min=float(HUValues.AIR_HU), max=float(HUValues.MAX_HU)
        )

        resampled = self._resample(volume_tensor, original_spacing)
        windowed = self._apply_lung_window(resampled)

        final_volume = windowed[0].numpy()
        return final_volume, final_volume.shape, original_spacing


    def _resample(self, volume_tensor: torch.Tensor,
                  original_spacing: Tuple[float, float, float]) -> torch.Tensor:
        """Resample to target isotropic spacing."""
        zoom_factors = [
            orig / target
            for orig, target in zip(original_spacing, self.config.target_spacing)
        ]

        zoomer = Zoom(
            zoom=zoom_factors, mode="bilinear",
            padding_mode="border", keep_size=False,
        )
        return zoomer(volume_tensor)


    def _apply_lung_window(self, volume_tensor: torch.Tensor) -> torch.Tensor:
        """Apply lung window and normalize intensity to [0, 1]."""
        window_center = getattr(self.config, 'window_center', PreProcessingConstants.WINDOW_CENTER)
        window_width = getattr(self.config, 'window_width', PreProcessingConstants.WINDOW_WIDTH)
        window_min = window_center - (window_width / 2.0)
        window_max = window_center + (window_width / 2.0)

        scaler = ScaleIntensityRange(
            a_min=window_min, a_max=window_max,
            b_min=IntensityRange.OUTPUT_MIN, b_max=IntensityRange.OUTPUT_MAX,
            clip=True,
        )
        return scaler(volume_tensor)


    def _clean_with_offset_detection(self, volume: np.ndarray,
                                      patient_id: str) -> np.ndarray:
        """Handle NaNs, detect padding/offset, clip to valid HU range."""
        
        nan_count = np.isnan(volume).sum()
        volume = np.nan_to_num(volume, nan=float(HUValues.AIR_HU)) if nan_count > 0 else volume
        self.logger.warning(
            f"[{patient_id}] Found {nan_count} NaN values, replacing with air HU"
        ) if nan_count > 0 else None

        padding_mask = volume < HUValues.PADDING_THRESHOLD
        padding_count = padding_mask.sum()

        center_slice = volume[volume.shape[0] // 2]
        valid_mask = (center_slice > HUValues.PADDING_THRESHOLD) & (center_slice < HUValues.VALID_PIXEL_MAX)
        valid_pixels = center_slice[valid_mask]

        needs_offset = (
            len(valid_pixels) > 0
            and np.percentile(valid_pixels, HUValues.OFFSET_PERCENTILE) > HUValues.OFFSET_THRESHOLD
        )

        if needs_offset:
            low_p = np.percentile(valid_pixels, HUValues.OFFSET_PERCENTILE)
            self.logger.info(
                f"[{patient_id}] Offset scan detected (P5={low_p:.1f}). "
                f"Applying -{HUValues.OFFSET_CORRECTION} correction."
            )
            volume[~padding_mask] -= float(HUValues.OFFSET_CORRECTION)

        volume[padding_mask] = float(HUValues.AIR_HU) if padding_count > 0 else volume[padding_mask]
        volume = np.clip(volume, float(HUValues.AIR_HU), float(HUValues.MAX_HU))
        return volume