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
        nodules = self._predict_nodule_features(detection_results)

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
        self, detection_results: list[SliceDetectionResult]
    ) -> list[Nodule]:
        # Pair each nodule with its parent slice's middle-channel image so we
        # can encode the full slice (with bbox) instead of just the crop.
        nodule_slice_pairs: list[tuple[DetectedNodule, np.ndarray]] = [
            (nodule, result.classifier_input.squeeze(0).numpy())
            for result in detection_results
            for nodule in result.nodules
        ]

        if not nodule_slice_pairs:
            return []

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
        """Encode the full CT slice as PNG with the nodule bounding box drawn on it.

        Args:
            slice_img: (H, W) float32 array in [0, 1] — middle channel of the 2.5D sandwich.
            detected:  nodule whose bbox coordinates are in absolute pixels of slice_img.
        """
        img_uint8 = (slice_img * 255).clip(0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)

        h, w = img_bgr.shape[:2]
        x1 = max(0, int(detected.x - detected.w / 2))
        y1 = max(0, int(detected.y - detected.h / 2))
        x2 = min(w - 1, int(detected.x + detected.w / 2))
        y2 = min(h - 1, int(detected.y + detected.h / 2))

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        _, encoded = cv2.imencode(".png", img_bgr)
        return base64.b64encode(encoded).decode("utf-8")

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
