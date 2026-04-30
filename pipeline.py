import logging
import base64
import uuid
from pathlib import Path
from typing import Protocol, runtime_checkable, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common.dto import PipelineResults, Nodule, BoundingBox, CancerClass
from DetectionModel.src.dto.nodule_features import NoduleFeatures
from DetectionModel.src.data_preprocessing.config import DataPrepConfig
from DetectionModel.src.data_preprocessing.sources.scan_adapters import DICOMScanSource
from DetectionModel.src.data_preprocessing.preprocessing.volume_processor import VolumePreprocessingPipeline
from DetectionModel.src.data_preprocessing.pipelines.inference_processor import InferencePipeline
from DetectionModel.src.data_preprocessing.preprocessing.slice_quality_gate import SliceQualityGate

logger = logging.getLogger(__name__)

CLASSIFICATION_INPUT_SIZE = (224, 224)
REGRESSION_INPUT_SIZE = (64, 64)
DEFAULT_MALIGNANCY_THRESHOLD = 3.0
DEFAULT_CONFIDENCE_THRESHOLD = 0.5


# ══════════════════════════════════════════════════════
#  Model Protocols — inject any implementation
# ══════════════════════════════════════════════════════

@runtime_checkable
class DetectionModelProtocol(Protocol):
    """Contract for YOLO-based nodule detection."""

    def eval(self) -> None: ...
    def predict_step(self, batch: tuple, batch_idx: int,
                     dataloader_idx: int = 0) -> list[dict]: ...


@runtime_checkable
class RegressionModelProtocol(Protocol):
    """Contract for nodule feature regression."""

    def predict_features(self, x: torch.Tensor) -> list[NoduleFeatures]: ...


@runtime_checkable
class ClassificationModelProtocol(Protocol):
    """Contract for cancer type classification."""

    def predict(self, images: np.ndarray) -> list[dict]: ...


# ══════════════════════════════════════════════════════
#  Main Inference Pipeline
# ══════════════════════════════════════════════════════

class MainPipeline:
    def __init__(self,
                 detection_model: DetectionModelProtocol,
                 regression_model: RegressionModelProtocol,
                 classification_model: ClassificationModelProtocol,
                 config: DataPrepConfig = None,
                 malignancy_threshold: float = DEFAULT_MALIGNANCY_THRESHOLD,
                 confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):

        self.detection_model = detection_model
        self.regression_model = regression_model
        self.classification_model = classification_model
        self.config = config or DataPrepConfig()
        self.malignancy_threshold = malignancy_threshold
        self.confidence_threshold = confidence_threshold

        self.volume_preprocessor = VolumePreprocessingPipeline(self.config)
        quality_gate = SliceQualityGate()
        self.inference_pipeline = InferencePipeline(self.config, quality_gate)

    # ──────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────

    def __call__(self, dicom_dir: Path) -> PipelineResults:
        patient_id = f"inf_{uuid.uuid4().hex[:8]}"

        # Stage 0 — Load DICOM volume
        volume_data = self._load_dicom_volume(dicom_dir, patient_id)
        if volume_data is None:
            return self._empty_results("Failed to load DICOM volume")

        # Stage 1 — Preprocess volume
        preprocess_result = self._preprocess_volume(volume_data, patient_id)
        if preprocess_result is None:
            return self._empty_results("Volume preprocessing failed")
        preprocessed_volume = preprocess_result[0]

        # Stage 2 — Detect nodules (YOLO)
        detection_results = self._detect_nodules(preprocessed_volume, patient_id)

        # Stage 3 — Predict nodule features (regression)
        nodules = self._predict_nodule_features(detection_results)

        # Stage 4 — Classify cancer type (only if thresholds exceeded)
        cancer_class = None
        if self._should_classify(nodules):
            cancer_class = self._classify_cancer(detection_results)
            logger.info(f"[{patient_id}] Classification triggered — "
                        f"threshold exceeded")
        else:
            logger.info(f"[{patient_id}] Classification skipped — "
                        f"no nodule exceeded thresholds")

        # Stage 5 — Assemble results
        total_detections = sum(len(r["nodules"]) for r in detection_results)
        notes = (f"Processed {len(detection_results)} slices, "
                 f"detected {total_detections} nodule(s)")
        if cancer_class is None and nodules:
            notes += " (classification skipped — below threshold)"

        return self._build_results(nodules, cancer_class, notes)

    # ──────────────────────────────────────────
    #  Stage 0 — DICOM Loading
    # ──────────────────────────────────────────

    def _load_dicom_volume(self, dicom_dir: Path, patient_id: str):
        try:
            source = DICOMScanSource(dicom_dir=dicom_dir,
                                     patient_id_override=patient_id)
            return source.load_volume()
        except Exception as e:
            logger.error(f"[{patient_id}] DICOM loading failed: {e}")
            return None

    # ──────────────────────────────────────────
    #  Stage 1 — Volume Preprocessing
    # ──────────────────────────────────────────

    def _preprocess_volume(self, volume_data, patient_id: str):
        return self.volume_preprocessor.preprocess(
            volume_data.volume, volume_data.spacing, patient_id
        )

    # ──────────────────────────────────────────
    #  Stage 2 — Nodule Detection (YOLO)
    # ──────────────────────────────────────────

    def _detect_nodules(self, volume: np.ndarray,
                        patient_id: str) -> list[dict]:
        processed_slices = self.inference_pipeline.prepare_slices_for_yolo(
            volume, patient_id
        )

        if not processed_slices:
            logger.warning(f"[{patient_id}] No slices passed quality gate")
            return []

        # Convert (H, W, 3) uint8 numpy → (3, H, W) float32 tensor
        tensors = []
        for ps in processed_slices:
            img = ps.enhanced_25d.astype(np.float32) / 255.0
            tensor = torch.from_numpy(img).permute(2, 0, 1)  # HWC → CHW
            tensors.append(tensor)

        batch = torch.stack(tensors)

        with torch.inference_mode():
            self.detection_model.eval()
            results = self.detection_model.predict_step(
                (batch, None), batch_idx=0
            )

        return results

    # ──────────────────────────────────────────
    #  Stage 3 — Nodule Feature Regression
    # ──────────────────────────────────────────

    def _predict_nodule_features(self, detection_results: list[dict]) -> list[Nodule]:
        all_nodules = [nodule for result in detection_results
                       for nodule in result["nodules"]]

        if not all_nodules:
            return []

        # Resize regression crops to (3, 64, 64)
        crops = []
        for nodule in all_nodules:
            crop = nodule["regression_input"]  # (3, crop_H, crop_W) tensor
            resized = F.interpolate(
                crop.unsqueeze(0).float(),
                size=REGRESSION_INPUT_SIZE,
                mode="bilinear",
                align_corners=False
            ).squeeze(0)
            crops.append(resized)

        crop_batch = torch.stack(crops)
        features_list = self.regression_model.predict_features(crop_batch)

        nodule_dtos = []
        for nodule_dict, features in zip(all_nodules, features_list):
            nodule_image_b64 = self._encode_nodule_image(
                nodule_dict["regression_input"]
            )
            nodule_dtos.append(Nodule(
                nodule_id=uuid.uuid4().hex[:8],
                bbox=BoundingBox(
                    x=nodule_dict["x"],
                    y=nodule_dict["y"],
                    height=nodule_dict["h"],
                    width=nodule_dict["w"],
                ),
                confidence=nodule_dict["confidence"],
                nodule_features=features,
                nodule_image=nodule_image_b64,
            ))

        return nodule_dtos

    # ──────────────────────────────────────────
    #  Threshold Gate
    # ──────────────────────────────────────────

    def _should_classify(self, nodules: list[Nodule]) -> bool:
        return any(
            n.confidence >= self.confidence_threshold
            and n.nodule_features.malignancy >= self.malignancy_threshold
            for n in nodules
        )

    # ──────────────────────────────────────────
    #  Stage 4 — Cancer Classification
    # ──────────────────────────────────────────

    def _classify_cancer(self, detection_results: list[dict]) -> Optional[CancerClass]:
        if not detection_results:
            return None

        images = []
        for result in detection_results:
            classifier_input = result["classifier_input"]  # (1, H, W) tensor
            slice_np = classifier_input.squeeze(0).numpy()  # (H, W)
            resized = cv2.resize(slice_np, CLASSIFICATION_INPUT_SIZE,
                                 interpolation=cv2.INTER_LINEAR)
            images.append(resized)

        # Stack into (N, 224, 224, 1) for Keras model
        batch = np.stack(images)[..., np.newaxis]

        predictions = self.classification_model.predict(batch)
        if not predictions:
            return None

        best = max(predictions, key=lambda p: p["confidence"])
        return CancerClass(
            cancer_type=best["cancer_type"],
            confidence=float(best["confidence"])
        )

    # ──────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────

    @staticmethod
    def _encode_nodule_image(crop_tensor: torch.Tensor) -> str:
        middle = crop_tensor[1].numpy()  # middle channel (H, W)
        img_uint8 = (middle * 255).clip(0, 255).astype(np.uint8)
        _, encoded = cv2.imencode(".png", img_uint8)
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
