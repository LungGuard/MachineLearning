import base64
import logging
import uuid
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common.constants import InferenceConstants
from common.dto import BoundingBox, CancerClass, Nodule, NoduleFeatures, PipelineResults
from common.model_protocols import *
from DetectionModel.src.data_preprocessing.config import DataPrepConfig
from DetectionModel.src.data_preprocessing.core.scan_protocols import (
    DetectedNodule,
    SliceDetectionResult,
)
from DetectionModel.src.data_preprocessing.pipelines.inference_processor import InferencePipeline
from DetectionModel.src.data_preprocessing.preprocessing.slice_quality_gate import SliceQualityGate
from DetectionModel.src.data_preprocessing.preprocessing.volume_processor import VolumePreprocessingPipeline
from DetectionModel.src.data_preprocessing.sources.scan_adapters import DICOMScanSource

logger = logging.getLogger(__name__)





class MainPipeline:
    def __init__(self,
                 detection_model: DetectionModelProtocol,
                 regression_model: RegressionModelProtocol,
                 classification_model: ClassificationModelProtocol,
                 config: Optional[DataPrepConfig] = None,
                 inference_constants: InferenceConstants = InferenceConstants(),
                 malignancy_threshold: Optional[float] = None,
                 confidence_threshold: Optional[float] = None):

        self.detection_model = detection_model
        self.regression_model = regression_model
        self.classification_model = classification_model
        self.config = config or DataPrepConfig()
        self.inference_constants = inference_constants
        self.malignancy_threshold = (
            malignancy_threshold
            if malignancy_threshold is not None
            else inference_constants.DEFAULT_MALIGNANCY_THRESHOLD
        )
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else inference_constants.DEFAULT_CONFIDENCE_THRESHOLD
        )

        self.volume_preprocessor = VolumePreprocessingPipeline(self.config)
        quality_gate = SliceQualityGate()
        self.inference_pipeline = InferencePipeline(self.config, quality_gate)

    def __call__(self, dicom_dir: Path) -> PipelineResults:
        patient_id = f"inf_{uuid.uuid4().hex[:8]}"

        volume_data = self._load_dicom_volume(dicom_dir, patient_id)
        if volume_data is None:
            return self._empty_results("Failed to load DICOM volume")

        preprocess_result = self._preprocess_volume(volume_data, patient_id)
        if preprocess_result is None:
            return self._empty_results("Volume preprocessing failed")
        preprocessed_volume = preprocess_result[0]

        detection_results = self._detect_nodules(preprocessed_volume, patient_id)
        nodules = self._predict_nodule_features(detection_results, patient_id)

        cancer_class: Optional[CancerClass] = None
        if self._should_classify(nodules):
            cancer_class = self._classify_cancer(detection_results)
            logger.info(f"[{patient_id}] Classification triggered — threshold exceeded")
        else:
            logger.info(f"[{patient_id}] Classification skipped — no nodule exceeded thresholds")

        total_detections = sum(len(r.nodules) for r in detection_results)
        notes = (
            f"Processed {len(detection_results)} slices, "
            f"detected {total_detections} nodule(s)"
        )
        if cancer_class is None and nodules:
            notes += " (classification skipped — below threshold)"

        return self._build_results(nodules, cancer_class, notes)

    def _load_dicom_volume(self, dicom_dir: Path, patient_id: str):
        try:
            source = DICOMScanSource(
                dicom_dir=dicom_dir, patient_id_override=patient_id
            )
            return source.load_volume()
        except Exception as e:
            logger.error(f"[{patient_id}] DICOM loading failed: {e}")
            return None

    def _preprocess_volume(self, volume_data, patient_id: str):
        return self.volume_preprocessor.preprocess(
            volume_data.volume, volume_data.spacing, patient_id
        )

    def _detect_nodules(self, volume: np.ndarray,
                        patient_id: str) -> list[SliceDetectionResult]:
        processed_slices = self.inference_pipeline.prepare_slices_for_yolo(
            volume, patient_id
        )

        if not processed_slices:
            logger.warning(f"[{patient_id}] No slices passed quality gate")
            return []

        batch_size = self.inference_constants.YOLO_INFERENCE_BATCH_SIZE
        all_results: list[SliceDetectionResult] = []

        with torch.inference_mode():
            self.detection_model.eval()
            for batch_idx, start in enumerate(range(0, len(processed_slices), batch_size)):
                chunk = processed_slices[start:start + batch_size]
                batch = self._build_yolo_batch(chunk)
                chunk_results = self.detection_model.predict_step(
                    (batch, None), batch_idx=batch_idx
                )
                all_results.extend(chunk_results)
                logger.debug(
                    f"[{patient_id}] YOLO mini-batch {batch_idx} "
                    f"({len(chunk)} slices): "
                    f"{sum(len(r.nodules) for r in chunk_results)} detection(s)"
                )

        return all_results

    def _build_yolo_batch(self, processed_slices) -> torch.Tensor:
        normalizer = self.inference_constants.YOLO_PIXEL_NORMALIZER
        tensors = [
            torch.from_numpy(
                ps.enhanced_25d.astype(np.float32) / normalizer
            ).permute(2, 0, 1)
            for ps in processed_slices
        ]
        return torch.stack(tensors)

    def _predict_nodule_features(
        self, detection_results: list[SliceDetectionResult], patient_id: str = ""
    ) -> list[Nodule]:
        # Pair each nodule with its parent slice's middle-channel image so we
        # can encode the slice (with bbox) instead of just the crop.
        nodule_slice_pairs: list[tuple[DetectedNodule, np.ndarray]] = [
            (nodule, result.classifier_input.squeeze(0).numpy())
            for result in detection_results
            for nodule in result.nodules
        ]

        if not nodule_slice_pairs:
            return []

        # Remove cross-slice duplicates: same physical nodule visible in adjacent slices.
        nodule_slice_pairs = self._cross_slice_nms(nodule_slice_pairs, patient_id=patient_id)

        all_nodules = [pair[0] for pair in nodule_slice_pairs]
        crop_batch = self._resize_regression_crops(all_nodules)
        features_list = self.regression_model.predict_features(crop_batch)

        return [
            self._build_nodule_dto(detected, features, slice_img)
            for (detected, slice_img), features in zip(nodule_slice_pairs, features_list)
        ]

    def _resize_regression_crops(
        self, detected_nodules: list[DetectedNodule]
    ) -> torch.Tensor:
        target = self.inference_constants.REGRESSION_INPUT_SIZE[0]
        return torch.stack([
            self._aspect_preserving_resize(dn.regression_input.float(), target)
            for dn in detected_nodules
        ])

    @staticmethod
    def _aspect_preserving_resize(crop: torch.Tensor, target: int) -> torch.Tensor:
        _, h, w = crop.shape
        scale = target / max(h, w)
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        resized = F.interpolate(
            crop.unsqueeze(0), size=(new_h, new_w),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
        pad_h = target - new_h
        pad_w = target - new_w
        pad_top    = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left   = pad_w // 2
        pad_right  = pad_w - pad_left
        return F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)

    def _build_nodule_dto(
        self, detected: DetectedNodule, features: NoduleFeatures, slice_img: np.ndarray
    ) -> Nodule:
        return Nodule(
            nodule_id=uuid.uuid4().hex[:8],
            bbox=BoundingBox(
                x=detected.x,
                y=detected.y,
                height=detected.h,
                width=detected.w,
            ),
            confidence=detected.confidence,
            nodule_features=features,
            nodule_image=self._encode_slice_with_bbox(slice_img, detected),
        )

    def _should_classify(self, nodules: list[Nodule]) -> bool:
        return any(
            n.confidence >= self.confidence_threshold
            and n.nodule_features.malignancy >= self.malignancy_threshold
            for n in nodules
        )

    def _classify_cancer(
        self, detection_results: list[SliceDetectionResult]
    ) -> Optional[CancerClass]:
        if not detection_results:
            return None

        target_size = self.inference_constants.CLASSIFICATION_INPUT_SIZE
        images = [
            cv2.resize(
                result.classifier_input.squeeze(0).numpy(),
                target_size,
                interpolation=cv2.INTER_LINEAR,
            )
            for result in detection_results
        ]

        batch = np.stack(images)[..., np.newaxis]

        if predictions := self.classification_model.predict(batch):
            return max(predictions, key=lambda p: p.confidence)
        else:
            return None

    @staticmethod
    def _encode_slice_with_bbox(slice_img: np.ndarray, detected: DetectedNodule) -> str:
        """Encode a context-cropped CT region as PNG with the nodule bounding box.

        The 2.5D sandwich used as YOLO input may have black zero-padding baked in
        (from center_crop_slice clamping MAX_CROP_SCALE).  We first detect the actual
        non-zero content bounds and restrict the crop to that region, so no artificial
        black columns bleed into the final image.

        Args:
            slice_img: (H, W) float32 array in [0, 1] — middle channel of the 2.5D sandwich.
            detected:  nodule whose bbox coordinates are in absolute pixels of slice_img.
        """
        _DISPLAY_MARGIN_FACTOR = 3.0   # context padding = 3× the bbox dimension on each side
        _DISPLAY_MIN_CONTEXT_MARGIN_PX = 80   # minimum margin in pixels regardless of bbox size
        _DISPLAY_OUTPUT_SIZE = 512     

        img_uint8 = (slice_img * 255).clip(0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)

        h, w = img_bgr.shape[:2]

        # Detect the non-zero content bbox so that zero-padding columns baked in by
        # center_crop_slice (when MAX_CROP_SCALE clamps the upscale) are excluded.
        content_ys, content_xs = np.where(slice_img > 1e-6)
        if content_ys.size > 0:
            cy1, cy2 = int(content_ys.min()), int(content_ys.max())
            cx1, cx2 = int(content_xs.min()), int(content_xs.max())
        else:
            cy1, cy2, cx1, cx2 = 0, h, 0, w   # degenerate guard: all-zero slice

        x1 = max(cx1, int(detected.x - detected.w / 2))
        y1 = max(cy1, int(detected.y - detected.h / 2))
        x2 = min(cx2, int(detected.x + detected.w / 2))
        y2 = min(cy2, int(detected.y + detected.h / 2))

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Crop a context region around the nodule, clamped to content bounds.
        margin_x = max(int((x2 - x1) * _DISPLAY_MARGIN_FACTOR), _DISPLAY_MIN_CONTEXT_MARGIN_PX)
        margin_y = max(int((y2 - y1) * _DISPLAY_MARGIN_FACTOR), _DISPLAY_MIN_CONTEXT_MARGIN_PX)
        crop_x1 = max(cx1, x1 - margin_x)
        crop_y1 = max(cy1, y1 - margin_y)
        crop_x2 = min(cx2, x2 + margin_x)
        crop_y2 = min(cy2, y2 + margin_y)

        # Force the crop to be square by expanding the shorter side within content bounds.
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1
        if crop_w > crop_h:
            expand = crop_w - crop_h
            crop_y1 = max(cy1, crop_y1 - expand // 2)
            crop_y2 = min(cy2, crop_y1 + crop_w)
        elif crop_h > crop_w:
            expand = crop_h - crop_w
            crop_x1 = max(cx1, crop_x1 - expand // 2)
            crop_x2 = min(cx2, crop_x1 + crop_h)

        crop = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]

        # Direct resize to output size — crop is square so no padding needed.
        output = cv2.resize(crop, (_DISPLAY_OUTPUT_SIZE, _DISPLAY_OUTPUT_SIZE),
                            interpolation=cv2.INTER_LINEAR)

        _, encoded = cv2.imencode(".png", output)
        return base64.b64encode(encoded).decode("utf-8")

    @staticmethod
    def _cross_slice_nms(
        pairs: list[tuple[DetectedNodule, np.ndarray]],
        patient_id: str = "",
        iomin_threshold: float = 0.5,
    ) -> list[tuple[DetectedNodule, np.ndarray]]:
        """Suppress cross-slice duplicates — same nodule detected in adjacent slices.

        Uses IoMin (intersection-over-min-area) ORed with a center-distance gate so
        that partial-volume end-slices (whose 2D bbox shrinks substantially relative to
        the middle slice) are still correctly merged with the main detection.

        Sorting by confidence descending ensures we keep the best detection of each lesion.
        """
        sorted_pairs = sorted(pairs, key=lambda p: p[0].confidence, reverse=True)
        kept: list[tuple[DetectedNodule, np.ndarray]] = []
        for nodule, img in sorted_pairs:
            if not any(
                MainPipeline._same_lesion(nodule, kept_nodule, iomin_threshold)
                for kept_nodule, _ in kept
            ):
                kept.append((nodule, img))

        logger.info(
            "[%s] Cross-slice NMS: %d raw detections → %d unique nodules",
            patient_id, len(pairs), len(kept),
        )
        return kept

    @staticmethod
    def _same_lesion(
        a: DetectedNodule, b: DetectedNodule, iomin_threshold: float
    ) -> bool:
        """True when two detections most likely represent the same physical nodule.

        Merges on IoMin ≥ iomin_threshold (robust to partial-volume size shrinkage)
        OR center distance < 0.5 × larger diameter (spatial proximity gate).
        Both pixel coords and distances are in mm because target_spacing = (1,1,1).
        """
        ax1, ay1 = a.x - a.w / 2, a.y - a.h / 2
        ax2, ay2 = a.x + a.w / 2, a.y + a.h / 2
        bx1, by1 = b.x - b.w / 2, b.y - b.h / 2
        bx2, by2 = b.x + b.w / 2, b.y + b.h / 2

        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)

        if ix2 > ix1 and iy2 > iy1:
            intersection = (ix2 - ix1) * (iy2 - iy1)
            min_area = min(a.w * a.h, b.w * b.h)
            if min_area > 0 and intersection / min_area >= iomin_threshold:
                return True

        # Center-distance gate: merge if centroids are within half the larger nodule's diameter.
        dist = ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5
        larger_diameter = max(max(a.w, a.h), max(b.w, b.h))
        return dist < 0.5 * larger_diameter

    @staticmethod
    def _build_results(nodules: list[Nodule],
                       cancer_class: Optional[CancerClass],
                       notes: str) -> PipelineResults:
        return PipelineResults(
            nodules=nodules,
            cancer_confidence=cancer_class.confidence if cancer_class else 0.0,
            cancer_class=cancer_class,
            notes=notes,
        )

    @staticmethod
    def _empty_results(notes: str) -> PipelineResults:
        return PipelineResults(
            nodules=[],
            cancer_confidence=0.0,
            cancer_class=None,
            notes=notes,
        )
