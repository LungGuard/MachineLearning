"""Inference Pipeline — In-memory slice processing for YOLO + CNN.

Handles the inference-specific flow:
  1. prepare_slices_for_yolo   → batch 2.5D slices for YOLO detection
  2. extract_nodule_crops      → crop detected nodules for CNN classifier
  3. extract_middle_slice      → pull center channel for single-slice CNN

All processing is in-memory — no disk I/O.
"""

import logging
from typing import List, Tuple, Optional

import cv2
import numpy as np

from ..core.scan_protocols import YOLODetection, ProcessedSlice, NoduleCropResult
from ..preprocessing.slice_quality_gate import SliceQualityGate

from ..preprocessing.slice_processor import SlicePreprocessor

logger = logging.getLogger(__name__)


class InferencePipeline:
    """In-memory slice processing for the inference (FastAPI) pathway.

    Depends on:
      - SliceQualityGate for CLAHE + validation
      - SlicePreprocessor for 2.5D sandwich creation
      - config for output_image_size, use_center_crop
    """

    def __init__(self, config, quality_gate: SliceQualityGate):
        self.config = config
        self.quality_gate = quality_gate
        self.logger = logging.getLogger(__name__)

    # ══════════════════════════════════════════
    #  Public API
    # ══════════════════════════════════════════

    def prepare_slice_for_inference(self, slice_idx: int, volume: np.ndarray,
                                     patient_id: str = "") -> ProcessedSlice:
        """Process one volume slice → in-memory ProcessedSlice."""
        context = f"inf_z{slice_idx:04d}"
        enhanced, _crop_info, passed, reason = self.prepare_slice_image(
            slice_idx, volume, patient_id, context
        )

        middle = (
            self.extract_middle_slice(enhanced)
            if enhanced is not None
            else np.array([])
        )

        return ProcessedSlice(
            slice_index=slice_idx,
            enhanced_25d=enhanced if enhanced is not None else np.array([]),
            middle_slice=middle,
            quality_passed=passed,
            reject_reason=reason,
        )

    def prepare_slices_for_yolo(self, volume: np.ndarray,
                                 patient_id: str = "",
                                 slice_indices: List[int] = None) -> List[ProcessedSlice]:
        """Batch-process volume slices for YOLO inference.

        Args:
            volume:        preprocessed volume (D, H, W)
            patient_id:    for logging
            slice_indices: specific slices to process (default: ALL axial slices)

        Returns:
            List of ProcessedSlice — only those that passed quality gate.
        """
        indices = slice_indices if slice_indices is not None else list(range(volume.shape[0]))

        all_slices = list(map(
            lambda idx: self.prepare_slice_for_inference(idx, volume, patient_id),
            indices
        ))

        passed_slices = list(filter(lambda s: s.quality_passed, all_slices))

        self.logger.info(
            f"[{patient_id}] YOLO prep: {len(passed_slices)}/{len(indices)} slices passed quality gate"
        )

        return passed_slices

    def extract_nodule_crops(self, detections: List[YOLODetection],
                              volume: np.ndarray,
                              patient_id: str = "",
                              crop_size: Tuple[int, int] = None) -> List[NoduleCropResult]:
        """Extract cropped nodule regions for the Stage 2 CNN classifier.

        For each YOLO detection:
          1. Build 2.5D sandwich at that slice (+ CLAHE)
          2. Convert YOLO normalized bbox → pixel coords
          3. Crop from both 2.5D and middle channel
          4. Resize crop to classifier input size if specified

        Args:
            detections: YOLO outputs from Stage 1
            volume:     preprocessed volume (D, H, W)
            patient_id: for logging
            crop_size:  (H, W) for classifier input. None = keep original crop.

        Returns:
            List[NoduleCropResult] with:
              .nodule_crop_single  → (crop_H, crop_W) grayscale for CNN
              .nodule_crop_25d     → (crop_H, crop_W, 3) for future use
        """
        results: List[NoduleCropResult] = []

        for det in detections:
            crop_result = self._extract_single_crop(det, volume, patient_id, crop_size)
            results.append(crop_result) if crop_result is not None else None

        self.logger.info(
            f"[{patient_id}] Nodule crops: {len(results)}/{len(detections)} "
            f"detections produced valid crops"
        )

        return results

    @staticmethod
    def extract_middle_slice(image_25d: np.ndarray) -> np.ndarray:
        """Extract the center channel from a 2.5D sandwich.

        Input:  (H, W, 3) — [slice-1, slice, slice+1]
        Output: (H, W)    — the actual center slice for CNN classifier
        """
        return (
            image_25d[:, :, image_25d.shape[2] // 2]
            if len(image_25d.shape) == 3
            else image_25d
        )

    # ══════════════════════════════════════════
    #  Shared Core — 2.5D + CLAHE (no I/O)
    #  Used by both InferencePipeline and CTScanProcessor
    # ══════════════════════════════════════════

    def prepare_slice_image(self, slice_idx: int, volume: np.ndarray,
                              patient_id: str,
                              context: str = "") -> Tuple[Optional[np.ndarray], Optional[tuple], bool, str]:
        """Create 2.5D sandwich → CLAHE + quality gate.

        Pure processing, no disk I/O, no bbox computation.

        Returns:
            (enhanced_25d_or_None, crop_info, quality_passed, reject_reason)
        """
        target_size = getattr(self.config, 'output_image_size', (512, 512))
        use_center_crop = getattr(self.config, 'use_center_crop', True)

        image_25d, crop_info = SlicePreprocessor.create_25d_sandwich(
            volume, slice_idx,
            target_size=target_size,
            use_center_crop=use_center_crop,
        )

        empty = image_25d is None or image_25d.size == 0
        enhanced, passed, reason = (
            (None, False, "EMPTY_SANDWICH")
            if empty
            else self.quality_gate.validate_and_enhance(
                image_25d, patient_id=patient_id, context=context
            )
        )

        return enhanced, crop_info, passed, reason

    # ══════════════════════════════════════════
    #  Private — Crop Extraction
    # ══════════════════════════════════════════

    def _extract_single_crop(self, detection: YOLODetection,
                              volume: np.ndarray, patient_id: str,
                              crop_size: Optional[Tuple[int, int]]) -> Optional[NoduleCropResult]:
        """Extract one nodule crop from a YOLO detection."""
        try:
            processed = self.prepare_slice_for_inference(
                detection.slice_index, volume, patient_id
            )

            if not processed.quality_passed:
                return None

            enhanced = processed.enhanced_25d
            img_h, img_w = enhanced.shape[0], enhanced.shape[1]

            # Convert YOLO normalized bbox → pixel coordinates
            cx, cy, bw, bh = detection.bbox_xywh_norm
            x1 = int(max(0, (cx - bw / 2) * img_w))
            y1 = int(max(0, (cy - bh / 2) * img_h))
            x2 = int(min(img_w, (cx + bw / 2) * img_w))
            y2 = int(min(img_h, (cy + bh / 2) * img_h))

            # Degenerate box guard
            valid_box = (x2 - x1) > 2 and (y2 - y1) > 2
            if not valid_box:
                return None

            # Crop from both channels
            crop_25d = enhanced[y1:y2, x1:x2, :]
            crop_single = processed.middle_slice[y1:y2, x1:x2]

            # Resize to classifier input if specified
            crop_25d_final, crop_single_final = (
                (
                    cv2.resize(crop_25d, (crop_size[1], crop_size[0]), interpolation=cv2.INTER_LINEAR),
                    cv2.resize(crop_single, (crop_size[1], crop_size[0]), interpolation=cv2.INTER_LINEAR),
                )
                if crop_size is not None
                else (crop_25d, crop_single)
            )

            return NoduleCropResult(
                detection=detection,
                full_slice_enhanced=enhanced,
                nodule_crop_single=crop_single_final,
                nodule_crop_25d=crop_25d_final,
            )
        except Exception as e:
            self.logger.error(
                f"[{patient_id}] Crop extraction failed for slice {detection.slice_index}: {e}"
            )
            return None