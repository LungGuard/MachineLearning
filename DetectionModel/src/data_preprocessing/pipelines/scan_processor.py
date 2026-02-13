"""CT Scan Processor — Thin Orchestrator.

Composes the extracted modules into a unified API:
─────────────────────────────────────────────────────────────────
  CTScanProcessor
    ├── VolumePreprocessingPipeline   (volume_preprocessing.py)
    ├── SliceQualityGate              (slice_quality.py)
    ├── InferencePipeline             (inference_pipeline.py)
    └── Data-prep logic               (this file)
─────────────────────────────────────────────────────────────────

Data-prep path:  process_scan() → process_nodule() → process_single_slice()
Inference path:  delegated entirely to self.inference.*
"""


import contextlib
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np

from constants.detection.dataset_constants import DatasetConstants

from ...utils import (
    VolumePreprocessor,
    BoundingBoxConverter,
)

from ..core.scan_protocols import ScanSource, VolumeData, NoduleData
from ..preprocessing.slice_quality_gate import SliceQualityGate, SliceQualityConfig
from ..preprocessing.volume_processor import VolumePreprocessingPipeline
from .inference_processor import InferencePipeline
from ..io.atomic_io import atomic_save_image_and_label

logger = logging.getLogger(__name__)


class CTScanProcessor:
    """Unified processor for both data preparation and inference.

    DATA-PREP:
        source = PyLIDCScanSource(scan, AnnotationProcessor)
        processor.process_scan(source, split='train')

    INFERENCE:
        source = DICOMScanSource(dicom_dir)
        vol, shape, spacing = processor.prepare_volume_from_source(source)
        slices = processor.inference.prepare_slices_for_yolo(vol, patient_id)
        crops  = processor.inference.extract_nodule_crops(detections, vol, patient_id)
        # crops[i].nodule_crop_single → CNN classifier
    """

    def __init__(self, config, directories: Dict[str, Path] = None,
                 quality_config: SliceQualityConfig = None):
        self.config = config
        self.directories = directories or {}
        self.logger = logging.getLogger(__name__)

        # ── Composed modules ──
        self._quality_gate = SliceQualityGate(quality_config)
        self._volume_pipeline = VolumePreprocessingPipeline(config)
        self.inference = InferencePipeline(config, self._quality_gate)

        self._stats = {
            "slices_generated": 0,
            "slices_rejected_quality": 0,
            "nodules_accepted": 0,
            "nodules_rejected_incomplete": 0,
        }

    @property
    def processing_stats(self) -> Dict:
        return dict(self._stats)

    # ══════════════════════════════════════════
    #  Shared — Volume Loading
    # ══════════════════════════════════════════

    def prepare_volume_from_source(self, source: ScanSource):
        """Load + preprocess volume. Single entry point for both pipelines.

        Returns: (volume, volume_shape, original_spacing) or None
        """
        volume_data = source.load_volume()
        return (
            self._volume_pipeline.preprocess(
                volume_data.volume, volume_data.spacing, source.patient_id
            )
            if volume_data is not None
            else None
        )

    # ══════════════════════════════════════════
    #  Data-Prep Pipeline — process + save to disk
    # ══════════════════════════════════════════

    def process_scan(self, source: ScanSource, split: str, pl_module=None):
        """Full data-preparation pipeline for one scan."""
        metadata_rows = []
        patient_id = source.patient_id

        volume_result = self.prepare_volume_from_source(source)
        if volume_result is None:
            return metadata_rows

        volume, volume_shape, original_spacing = volume_result

        nodules = source.extract_nodules(
            volume_shape, original_spacing, self.config.target_spacing
        )

        valid_nodules = list(filter(
            lambda n: self._is_valid_nodule(n.features), nodules
        ))

        for nodule in valid_nodules:
            rows = self._process_nodule(
                nodule, patient_id, split, volume, volume_shape
            )
            metadata_rows.extend(rows)

        self.logger.info(
            f"[{patient_id}] Complete: {len(metadata_rows)} slices | Stats: {self._stats}"
        )
        return metadata_rows

    def _process_nodule(self, nodule: NoduleData, patient_id: str, split: str,
                         volume: np.ndarray, volume_shape: Tuple) -> List[Dict]:
        """Process all slices for one nodule with atomic guarantee."""
        try:
            required = self.config.slices_per_nodule
            candidates = self._select_candidate_slices(nodule.slice_indices, required)

            accepted: List[Dict] = []
            for slice_idx in candidates:
                remaining = required - len(accepted)
                metadata = (
                    self._process_single_slice(
                        slice_idx, nodule, volume, volume_shape, patient_id, split
                    )
                    if remaining > 0
                    else None
                )
                accepted.append(metadata) if metadata is not None else None

            return self._enforce_nodule_integrity(
                accepted, required, patient_id, nodule.index
            )
        except Exception as e:
            self.logger.error(f"[{patient_id}] Nodule error: {e}")
            return []

    def _process_single_slice(self, slice_idx: int, nodule: NoduleData,
                               volume: np.ndarray, volume_shape: Tuple,
                               patient_id: str, split: str) -> Optional[Dict]:
        """Data-prep: shared core → bbox → save to disk → metadata."""
        try:
            context = f"n{nodule.index:02d}_z{slice_idx:04d}"

            # ── Shared core (reuses inference pipeline's internal method) ──
            enhanced, crop_info, passed, reason = self.inference._prepare_slice_image(
                slice_idx, volume, patient_id, context
            )

            if not passed:
                self._stats["slices_rejected_quality"] += 1
                return None

            # ── Bbox computation (data-prep only) ──
            target_size = getattr(self.config, 'output_image_size', (512, 512))
            features = nodule.features

            bbox = BoundingBoxConverter.compute_nodule_bbox_yolo(
                nodule.centroid_zyx,
                features[DatasetConstants.Features.FEATURE_DIAMETER_MM],
                volume_shape,
                self.config.target_spacing,
                self.config.bbox_padding_factor,
            )

            if bbox is None:
                return None

            bbox = self._adjust_bbox_for_output(bbox, volume_shape, target_size, crop_info)

            if bbox is None:
                return None

            # ── Save to disk (data-prep only) ──
            filename = f"{patient_id}_n{nodule.index:02d}_z{slice_idx:04d}"
            image_dir = self.directories[f'{split}_images']
            label_dir = self.directories[f'{split}_labels']

            save_result = atomic_save_image_and_label(
                enhanced, bbox, self.config.class_id,
                image_dir / f"{filename}.jpg",
                label_dir / f"{filename}.txt",
            )

            if not save_result.success:
                return None

            self._stats["slices_generated"] += 1

            return self._build_metadata(
                filename,
                patient_id,
                split,
                nodule,
                slice_idx,
                bbox,
                save_result,
                volume_shape,
            )
        except Exception as e:
            self.logger.error(f"[{patient_id}] Slice processing error: {e}")
            self.logger.debug(traceback.format_exc())
            return None

    # ══════════════════════════════════════════
    #  Private Helpers
    # ══════════════════════════════════════════

    def _is_valid_nodule(self, features: Dict) -> bool:
        diameter = features.get(DatasetConstants.Features.FEATURE_DIAMETER_MM, 0)
        annot_count = features.get(DatasetConstants.Features.FEATURE_ANNOTATION_COUNT, 0)
        diameter_valid = (
            self.config.min_nodule_diameter <= diameter <= self.config.max_nodule_diameter
        )
        return diameter_valid and annot_count > 0

    @staticmethod
    def _adjust_bbox_for_output(bbox, volume_shape, target_size, crop_info):
        result = bbox
        if target_size is not None:
            original_size = (volume_shape[1], volume_shape[2])
            result = (
                BoundingBoxConverter.adjust_bbox_for_center_crop(
                    bbox, original_size, target_size,
                    crop_info[0], crop_info[1],
                )
                if crop_info is not None
                else BoundingBoxConverter.adjust_bbox_for_resize(
                    bbox, original_size, target_size, preserve_aspect_ratio=True,
                )
            )
        return result

    def _enforce_nodule_integrity(self, results: List[Dict], required: int,
                                   patient_id: str, nodule_idx: int) -> List[Dict]:
        """All-or-nothing: return exactly `required` slices, or clean up."""
        have_enough = len(results) >= required
        accepted = results[:required] if have_enough else []

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
        for entry in metadata_entries:
            with contextlib.suppress(Exception):
                img_path = Path(entry.get(DatasetConstants.IMAGE_PATH, ""))
                lbl_path = Path(entry.get(DatasetConstants.LABEL_PATH, ""))
                img_path.unlink() if img_path.exists() else None
                lbl_path.unlink() if lbl_path.exists() else None

    @staticmethod
    def _select_candidate_slices(slice_indices: List[int], num_slices: int) -> List[int]:
        """Select up to 2× candidates, evenly distributed."""
        total = len(slice_indices)
        num_candidates = min(total, num_slices * 2)
        return (
            []
            if total == 0
            else (
                slice_indices
                if total <= num_candidates
                else (
                    [
                        slice_indices[int(i * (total - 1) / (num_candidates - 1))]
                        for i in range(num_candidates)
                    ]
                    if num_candidates > 1
                    else [slice_indices[total // 2]]
                )
            )
        )

    @staticmethod
    def _build_metadata(filename, patient_id, split, nodule: NoduleData,
                         slice_idx, bbox, save_result, volume_shape) -> Dict:
        features = nodule.features
        centroid = nodule.centroid_zyx
        F = DatasetConstants.Features
        C = DatasetConstants.CENTROID
        B = DatasetConstants.BBOX
        V = DatasetConstants.VOLUME

        return {
            DatasetConstants.FILE_NAME: filename,
            DatasetConstants.PATIENT_ID: patient_id,
            DatasetConstants.SPLIT_GROUP: split,
            DatasetConstants.NOUDLE_INDEX: nodule.index,
            DatasetConstants.SLICE_INDEX: slice_idx,
            F.FEATURE_DIAMETER_MM: features[F.FEATURE_DIAMETER_MM],
            F.FEATURE_MALIGNANCY: features[F.FEATURE_MALIGNANCY],
            F.FEATURE_SPICULATION: features[F.FEATURE_SPICULATION],
            F.FEATURE_LOBULATION: features[F.FEATURE_LOBULATION],
            F.FEATURE_SUBTLETY: features[F.FEATURE_SUBTLETY],
            F.FEATURE_SPHERICITY: features[F.FEATURE_SPHERICITY],
            F.FEATURE_MARGIN: features[F.FEATURE_MARGIN],
            F.FEATURE_TEXTURE: features[F.FEATURE_TEXTURE],
            F.FEATURE_CALCIFICATION: features[F.FEATURE_CALCIFICATION],
            F.FEATURE_INTERNAL_STRUCTURE: features[F.FEATURE_INTERNAL_STRUCTURE],
            F.FEATURE_ANNOTATION_COUNT: features[F.FEATURE_ANNOTATION_COUNT],
            C.CENTROID_Z: centroid[0],
            C.CENTROID_Y: centroid[1],
            C.CENTROID_X: centroid[2],
            B.BBOX_X: bbox[0],
            B.BBOX_Y: bbox[1],
            B.BBOX_W: bbox[2],
            B.BBOX_H: bbox[3],
            DatasetConstants.IMAGE_PATH: save_result.image_path,
            DatasetConstants.LABEL_PATH: save_result.label_path,
            V.VOLUME_DEPTH: volume_shape[0],
            V.VOLUME_HEIGHT: volume_shape[1],
            V.VOLUME_WIDTH: volume_shape[2],
        }