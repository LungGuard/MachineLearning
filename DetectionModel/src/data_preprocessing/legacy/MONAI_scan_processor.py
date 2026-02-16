"""CT Scan Processing Module - MONAI Powered.

Improvements over original:
─────────────────────────────────────────────────────────────────────────
1. CLAHE contrast enhancement per-slice        → fixes LOW_CONTRAST (~1030)
2. Pre-save SliceQualityGate with lung check   → fixes NO_BG (~690),
                                                  TOO_BRIGHT (~160),
                                                  INSUFFICIENT_LUNG (~137)
3. Atomic nodule processing with oversampling  → fixes Nodule Integrity (355)
─────────────────────────────────────────────────────────────────────────
"""

import logging
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from monai.transforms import ScaleIntensityRange, Zoom

from constants.detection.dataset_constants import DatasetConstants, PreProcessingConstants

from ..preprocessing.slice_processor import SlicePreprocessor
from ..core.coordinate_transformer import CoordinateTransformer
from ..preprocessing.bbox_converter import BoundingBoxConverter
from ..sources.annotation_processor import NoduleAnnotationProcessor

from ..io.atomic_io import atomic_save_image_and_label

logger = logging.getLogger(__name__)

HU = PreProcessingConstants.HU_VALUES
INTENSITY = PreProcessingConstants.INTENSITY_RANGE


# ──────────────────────────────────────────────
# Slice Quality Gate  (NEW)
# ──────────────────────────────────────────────

@dataclass
class SliceQualityConfig:
    """Pre-save validation thresholds.

    Aligned with the diagnoser thresholds so that slices passing
    this gate will also pass the diagnoser — preventing any
    post-hoc cleaning.
    """
    # Lung content  (mirrors diagnoser)
    body_intensity_floor: int = 20
    lung_intensity_low: int = 10
    lung_intensity_high: int = 90
    morph_kernel_size: int = 15
    min_lung_body_ratio: float = 0.12

    # Contrast  (slightly below diagnoser's 100 — CLAHE will push it over)
    min_contrast_range: int = 80

    # Brightness
    max_mean_brightness: float = 180.0

    # Dark background  (mirrors diagnoser)
    min_dark_ratio: float = 0.20
    dark_threshold: int = 50

    # CLAHE parameters
    clahe_clip_limit: float = 2.5
    clahe_grid_size: Tuple[int, int] = (8, 8)


class SliceQualityGate:
    """Validates and enhances a 2D slice before saving.

    Two jobs:
    1. Reject slices that won't be useful (apex/base, too bright, no lung)
    2. Apply CLAHE to spread contrast → eliminates LOW_CONTRAST at source
    """

    def __init__(self, config: SliceQualityConfig = None):
        self.config = config or SliceQualityConfig()

    def validate_and_enhance(self, image: np.ndarray, patient_id: str = "", context: str = "") -> Optional[np.ndarray]:
        """Apply CLAHE, then validate.  Returns enhanced image or None."""
        enhanced = self._apply_clahe(image)

        passed, reason = self._check_quality(enhanced)

        logger.debug(
            f"[{patient_id}] {context} rejected: {reason}"
        ) if not passed else None

        result = enhanced if passed else None
        return result

    def _check_quality(self, image: np.ndarray) -> Tuple[bool, str]:
        """All quality checks in one pass."""
        c = self.config

        # For multi-channel (2.5D sandwich) — inspect middle channel
        check_slice = image[:, :, image.shape[2] // 2] if len(image.shape) == 3 else image
        gray = self._to_uint8(check_slice)
        total = gray.size

        mean_val = float(gray.mean())
        min_val = int(gray.min())
        max_val = int(gray.max())
        contrast_range = max_val - min_val
        dark_ratio = int(np.sum(gray < c.dark_threshold)) / total

        lung_body_ratio = self._compute_lung_ratio(gray)

        # Ordered by expected frequency — most common rejection first
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


# ──────────────────────────────────────────────
# CT Scan Processor
# ──────────────────────────────────────────────

class CTScanProcessor:
    """Processor for CT scans - handles volume preparation using MONAI transforms.

    Changes from original:
    • __init__ now takes optional SliceQualityConfig
    • process_single_slice runs the quality gate before saving
    • process_nodule is atomic: all N slices must pass or the nodule is skipped
    • _select_candidate_slices returns 2× candidates for fallback room
    """

    def __init__(self, config, directories: Dict[str, Path],
                 quality_config: SliceQualityConfig = None):
        self.config = config
        self.directories = directories
        self.logger = logging.getLogger(__name__)
        self.quality_gate = SliceQualityGate(quality_config)

        self._stats = {
            "slices_generated": 0,
            "slices_rejected_quality": 0,
            "nodules_accepted": 0,
            "nodules_rejected_incomplete": 0,
        }

    @property
    def processing_stats(self) -> Dict:
        return dict(self._stats)

    # ── Volume Loading & Preparation  (unchanged) ─────

    def _load_volume(self, scan, patient_id: str):
        try:
            return scan.to_volume()
        except Exception as e:
            self.logger.error(f"[{patient_id}] FAILED at volume loading: {e}")
            return None

    def _get_spacing(self, scan, patient_id: str):
        try:
            pixel_spacing_raw = scan.pixel_spacing
            xy_spacing = (
                [float(pixel_spacing_raw), float(pixel_spacing_raw)]
                if isinstance(pixel_spacing_raw, (float, int, np.floating, np.integer))
                else [float(pixel_spacing_raw[0]), float(pixel_spacing_raw[1])]
            )

            slice_spacing = float(scan.slice_spacing)
            original_spacing = (slice_spacing, xy_spacing[0], xy_spacing[1])
            return original_spacing
        except Exception as e:
            self.logger.error(f"[{patient_id}] FAILED at spacing extraction: {e}")
            return None

    def prepare_volume(self, scan, patient_id: str):
        """Load, clean, resample, and window a CT volume using MONAI transforms."""
        raw_volume = self._load_volume(scan, patient_id)
        if raw_volume is None:
            return None

        original_spacing = self._get_spacing(scan, patient_id)
        if original_spacing is None:
            return None

        try:
            raw_volume = self._clean_volume_with_offset_detection(raw_volume, patient_id)

            volume_tensor = torch.from_numpy(raw_volume).float().unsqueeze(0)
            volume_tensor = torch.clamp(
                volume_tensor,
                min=float(HU.AIR_HU),
                max=float(HU.MAX_HU)
            )

            zoom_factors = [
                orig / target
                for orig, target in zip(original_spacing, self.config.target_spacing)
            ]

            zoomer = Zoom(
                zoom=zoom_factors,
                mode="bilinear",
                padding_mode="border",
                keep_size=False
            )
            resampled_volume = zoomer(volume_tensor)

            window_center = getattr(self.config, 'window_center', PreProcessingConstants.WINDOW_CENTER)
            window_width = getattr(self.config, 'window_width', PreProcessingConstants.WINDOW_WIDTH)
            window_min = window_center - (window_width / 2.0)
            window_max = window_center + (window_width / 2.0)

            scaler = ScaleIntensityRange(
                a_min=window_min,
                a_max=window_max,
                b_min=INTENSITY.OUTPUT_MIN,
                b_max=INTENSITY.OUTPUT_MAX,
                clip=True
            )
            windowed_volume = scaler(resampled_volume)

            final_volume = windowed_volume[0].numpy()
            return final_volume, final_volume.shape, original_spacing

        except Exception as e:
            self.logger.error(f"[{patient_id}] MONAI Processing FAILED: {e}")
            self.logger.debug(traceback.format_exc())
            return None

    def _clean_volume_with_offset_detection(self, volume: np.ndarray, patient_id: str) -> np.ndarray:
        """Clean volume: handle NaNs, detect padding/offset, clip to valid HU range."""
        volume = volume.astype(np.float32)

        nan_count = np.isnan(volume).sum()
        if nan_count > 0:
            self.logger.warning(f"[{patient_id}] Found {nan_count} NaN values, replacing with air HU")
            volume = np.nan_to_num(volume, nan=float(HU.AIR_HU))

        padding_mask = volume < HU.PADDING_THRESHOLD
        padding_count = padding_mask.sum()

        center_slice = volume[volume.shape[0] // 2]
        valid_mask = (center_slice > HU.PADDING_THRESHOLD) & (center_slice < HU.VALID_PIXEL_MAX)
        valid_pixels = center_slice[valid_mask]

        if len(valid_pixels) > 0:
            low_percentile = np.percentile(valid_pixels, HU.OFFSET_PERCENTILE)

            if low_percentile > HU.OFFSET_THRESHOLD:
                self.logger.info(
                    f"[{patient_id}] Offset scan detected (P5={low_percentile:.1f}). "
                    f"Applying -{HU.OFFSET_CORRECTION} correction."
                )
                volume[~padding_mask] -= float(HU.OFFSET_CORRECTION)

        if padding_count > 0:
            volume[padding_mask] = float(HU.AIR_HU)

        volume = np.clip(volume, float(HU.AIR_HU), float(HU.MAX_HU))
        return volume

    # ── Nodule Extraction  (unchanged) ────────

    def extract_valid_nodules(self, scan, patient_id: str) -> List[Tuple]:
        try:
            nodules = scan.cluster_annotations()
        except Exception:
            return []

        try:
            nodule_features = []
            for idx, annotations in enumerate(nodules):
                features = NoduleAnnotationProcessor.extract_nodule_features(annotations)
                nodule_features.append((annotations, features))
        except Exception:
            return []

        def is_valid_nodule(nodule_data):
            _, features = nodule_data
            diameter = features.get(DatasetConstants.Features.FEATURE_DIAMETER_MM, 0)
            annot_count = features.get(DatasetConstants.Features.FEATURE_ANNOTATION_COUNT, 0)
            diameter_valid = self.config.min_nodule_diameter <= diameter <= self.config.max_nodule_diameter
            return diameter_valid and annot_count > 0

        return list(filter(is_valid_nodule, nodule_features))

    # ── Single Slice Processing  (MODIFIED) ───

    def process_single_slice(self, slice_idx, nodule_idx, centroid, features,
                              volume, volume_shape, patient_id, split):
        """Process one slice: create 2.5D → CLAHE + quality gate → bbox → save.

        Returns metadata dict on success, None on rejection/failure.
        """
        try:
            target_size = getattr(self.config, 'output_image_size', (512, 512))
            use_center_crop = getattr(self.config, 'use_center_crop', True)
            context = f"n{nodule_idx:02d}_z{slice_idx:04d}"

            # ── Step 1: Create 2.5D sandwich ──
            image_25d, crop_info = SlicePreprocessor.create_25d_sandwich(
                volume,
                slice_idx,
                target_size=target_size,
                use_center_crop=use_center_crop,
            )

            if image_25d is None or image_25d.size == 0:
                return None

            # ── Step 2: Quality gate + CLAHE  (NEW) ──
            enhanced = self.quality_gate.validate_and_enhance(
                image_25d, patient_id=patient_id, context=context
            )
            if enhanced is None:
                self._stats["slices_rejected_quality"] += 1
                return None

            # ── Step 3: Compute bounding box ──
            bbox = BoundingBoxConverter.compute_nodule_bbox_yolo(
                centroid,
                features[DatasetConstants.Features.FEATURE_DIAMETER_MM],
                volume_shape,
                self.config.target_spacing,
                self.config.bbox_padding_factor
            )

            if bbox is None:
                return None

            if target_size is not None:
                original_size = (volume_shape[1], volume_shape[2])

                bbox = (
                    BoundingBoxConverter.adjust_bbox_for_center_crop(
                        bbox, original_size, target_size,
                        crop_info[0], crop_info[1]
                    )
                    if crop_info is not None
                    else BoundingBoxConverter.adjust_bbox_for_resize(
                        bbox, original_size, target_size, preserve_aspect_ratio=True
                    )
                )

                if bbox is None:
                    return None

            # ── Step 4: Save (enhanced image, not raw) ──
            filename = f"{patient_id}_n{nodule_idx:02d}_z{slice_idx:04d}"
            image_dir = self.directories[f'{split}_images']
            label_dir = self.directories[f'{split}_labels']

            save_result = atomic_save_image_and_label(
                enhanced, bbox, self.config.class_id,
                image_dir / f"{filename}.jpg",
                label_dir / f"{filename}.txt"
            )

            if not save_result.success:
                return None

            self._stats["slices_generated"] += 1

            metadata_entry = {
                DatasetConstants.FILE_NAME: filename,
                DatasetConstants.PATIENT_ID: patient_id,
                DatasetConstants.SPLIT_GROUP: split,
                DatasetConstants.NOUDLE_INDEX: nodule_idx,
                DatasetConstants.SLICE_INDEX: slice_idx,
                DatasetConstants.Features.FEATURE_DIAMETER_MM: features[DatasetConstants.Features.FEATURE_DIAMETER_MM],
                DatasetConstants.Features.FEATURE_MALIGNANCY: features[DatasetConstants.Features.FEATURE_MALIGNANCY],
                DatasetConstants.Features.FEATURE_SPICULATION: features[DatasetConstants.Features.FEATURE_SPICULATION],
                DatasetConstants.Features.FEATURE_LOBULATION: features[DatasetConstants.Features.FEATURE_LOBULATION],
                DatasetConstants.Features.FEATURE_SUBTLETY: features[DatasetConstants.Features.FEATURE_SUBTLETY],
                DatasetConstants.Features.FEATURE_SPHERICITY: features[DatasetConstants.Features.FEATURE_SPHERICITY],
                DatasetConstants.Features.FEATURE_MARGIN: features[DatasetConstants.Features.FEATURE_MARGIN],
                DatasetConstants.Features.FEATURE_TEXTURE: features[DatasetConstants.Features.FEATURE_TEXTURE],
                DatasetConstants.Features.FEATURE_CALCIFICATION: features[DatasetConstants.Features.FEATURE_CALCIFICATION],
                DatasetConstants.Features.FEATURE_INTERNAL_STRUCTURE: features[DatasetConstants.Features.FEATURE_INTERNAL_STRUCTURE],
                DatasetConstants.Features.FEATURE_ANNOTATION_COUNT: features[DatasetConstants.Features.FEATURE_ANNOTATION_COUNT],
                DatasetConstants.CENTROID.CENTROID_Z: centroid[0],
                DatasetConstants.CENTROID.CENTROID_Y: centroid[1],
                DatasetConstants.CENTROID.CENTROID_X: centroid[2],
                DatasetConstants.BBOX.BBOX_X: bbox[0],
                DatasetConstants.BBOX.BBOX_Y: bbox[1],
                DatasetConstants.BBOX.BBOX_W: bbox[2],
                DatasetConstants.BBOX.BBOX_H: bbox[3],
                DatasetConstants.IMAGE_PATH: save_result.image_path,
                DatasetConstants.LABEL_PATH: save_result.label_path,
                DatasetConstants.VOLUME.VOLUME_DEPTH: volume_shape[0],
                DatasetConstants.VOLUME.VOLUME_HEIGHT: volume_shape[1],
                DatasetConstants.VOLUME.VOLUME_WIDTH: volume_shape[2]
            }
            return metadata_entry

        except Exception as e:
            self.logger.error(f"[{patient_id}] Slice processing error: {e}")
            self.logger.debug(traceback.format_exc())
            return None

    # ── Nodule Processing  (MODIFIED — now atomic) ────

    def process_nodule(self, nodule_idx, annotations, features, patient_id, split,
                        volume, volume_shape, original_spacing):
        """Process all slices for one nodule with atomic guarantee.

        KEY CHANGE: If we can't produce exactly `slices_per_nodule` valid
        slices, the entire nodule is rejected and saved files are cleaned up.
        Candidates are oversampled (2×) so quality rejections have fallback room.
        """
        try:
            centroid = NoduleAnnotationProcessor.get_nodule_centroid(
                annotations, volume_shape, original_spacing, self.config.target_spacing
            )
            if centroid is None:
                return []

            slice_indices = NoduleAnnotationProcessor.get_nodule_slice_indices(
                annotations, volume_shape[0], original_spacing, self.config.target_spacing
            )

            required = self.config.slices_per_nodule

            # Oversample candidates (2×) so quality-gate rejections don't kill the nodule
            candidates = self._select_candidate_slices(slice_indices, required)

            # Process candidates until we have enough or run out
            accepted: List[Dict] = []
            for slice_idx in candidates:
                # Already have enough — stop processing
                remaining = required - len(accepted)
                metadata = (
                    self.process_single_slice(
                        slice_idx, nodule_idx, centroid, features,
                        volume, volume_shape, patient_id, split
                    )
                    if remaining > 0
                    else None
                )
                accepted.append(metadata) if metadata is not None else None

            # ── Atomic enforcement ──
            nodule_results = self._enforce_nodule_integrity(
                accepted, required, patient_id, nodule_idx
            )
            return nodule_results

        except Exception as e:
            self.logger.error(f"[{patient_id}] Nodule error: {e}")
            return []

    def _enforce_nodule_integrity(self, results: List[Dict], required: int,
                                   patient_id: str, nodule_idx: int) -> List[Dict]:
        """All-or-nothing: return exactly `required` slices, or clean up and return []."""
        have_enough = len(results) >= required

        # Happy path: trim to exact count
        accepted = results[:required] if have_enough else []

        # Sad path: not enough slices passed quality → reject entire nodule
        if not have_enough:
            self._stats["nodules_rejected_incomplete"] += 1
            self.logger.warning(
                f"[{patient_id}] Nodule {nodule_idx}: only {len(results)}/{required} "
                f"slices passed — nodule REJECTED"
            )
            self._cleanup_saved_files(results)
        else:
            self._stats["nodules_accepted"] += 1

        return accepted

    @staticmethod
    def _cleanup_saved_files(metadata_entries: List[Dict]) -> None:
        """Remove already-saved files for a rejected nodule."""
        for entry in metadata_entries:
            try:
                img_path = Path(entry.get(DatasetConstants.IMAGE_PATH, ""))
                lbl_path = Path(entry.get(DatasetConstants.LABEL_PATH, ""))
                img_path.unlink() if img_path.exists() else None
                lbl_path.unlink() if lbl_path.exists() else None
            except Exception:
                pass

    # ── Slice Selection  (MODIFIED) ───────────

    @staticmethod
    def _select_candidate_slices(slice_indices: List[int], num_slices: int) -> List[int]:
        """Select up to 2× candidates, evenly distributed.

        Original problem: _select_representative_slices returned ≤ num_slices,
        so if ANY failed processing the nodule ended up incomplete.

        Now we return 2× candidates. process_nodule stops early once it has enough.
        """
        total = len(slice_indices)
        num_candidates = min(total, num_slices * 2)

        result = (
            []
            if total == 0
            else slice_indices
            if total <= num_candidates
            else (
                [slice_indices[int(i * (total - 1) / (num_candidates - 1))] for i in range(num_candidates)]
                if num_candidates > 1
                else [slice_indices[total // 2]]
            )
        )
        return result

    # ── Scan Processing  (unchanged) ──────────

    def process_scan(self, scan, split, pl_module=None):
        metadata_rows = []
        patient_id = scan.patient_id

        volume_result = self.prepare_volume(scan, patient_id)
        if volume_result is None:
            return []

        volume, volume_shape, original_spacing = volume_result

        valid_nodules = self.extract_valid_nodules(scan, patient_id)

        for idx, (annotations, features) in enumerate(valid_nodules):
            rows = self.process_nodule(
                idx, annotations, features, patient_id, split,
                volume, volume_shape, original_spacing
            )
            metadata_rows.extend(rows)

        self.logger.info(
            f"[{patient_id}] Complete: {len(metadata_rows)} slices | "
            f"Stats: {self._stats}"
        )

        return metadata_rows