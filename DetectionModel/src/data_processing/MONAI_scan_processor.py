"""CT Scan Processing Module - MONAI Powered."""

import logging
import traceback
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

import torch
from monai.transforms import ScaleIntensityRange, Zoom

from constants.detection.dataset_constants import RegModelConstants, PreProcessingConstants

from ..utils import (
    VolumePreprocessor,
    CoordinateTransformer,
    BoundingBoxConverter,
    NoduleAnnotationProcessor
)

from .file_io import atomic_save_image_and_label

logger = logging.getLogger(__name__)

HU = PreProcessingConstants.HU_VALUES
INTENSITY = PreProcessingConstants.INTENSITY_RANGE


class CTScanProcessor:
    """Processor for CT scans - handles volume preparation using MONAI transforms."""

    def __init__(self, config, directories: Dict[str, Path]):
        self.config = config
        self.directories = directories
        self.logger = logging.getLogger(__name__)

    def _load_volume(self, scan, patient_id: str):
        try:
            return scan.to_volume()
        except Exception as e:
            self.logger.error(f"[{patient_id}] FAILED at volume loading: {e}")
            return None

    def _get_spacing(self, scan, patient_id: str):
        try:
            pixel_spacing_raw = scan.pixel_spacing
            if isinstance(pixel_spacing_raw, (float, int, np.floating, np.integer)):
                 xy_spacing = [float(pixel_spacing_raw), float(pixel_spacing_raw)]
            else:
                 xy_spacing = [float(pixel_spacing_raw[0]), float(pixel_spacing_raw[1])]

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
            diameter = features.get(RegModelConstants.Features.FEATURE_DIAMETER_MM, 0)
            annot_count = features.get(RegModelConstants.Features.FEATURE_ANNOTATION_COUNT, 0)
            diameter_valid = self.config.min_nodule_diameter <= diameter <= self.config.max_nodule_diameter
            return diameter_valid and annot_count > 0

        return list(filter(is_valid_nodule, nodule_features))

    def process_single_slice(self, slice_idx, nodule_idx, centroid, features, volume, volume_shape, patient_id, split):
        try:
            target_size = getattr(self.config, 'output_image_size', (512, 512))
            use_center_crop = getattr(self.config, 'use_center_crop', True)

            image_25d, crop_info = VolumePreprocessor.create_25d_sandwich(
                volume,
                slice_idx,
                target_size=target_size,
                use_center_crop=use_center_crop,
            )

            if image_25d is None or image_25d.size == 0:
                return None

            bbox = BoundingBoxConverter.compute_nodule_bbox_yolo(
                centroid,
                features[RegModelConstants.Features.FEATURE_DIAMETER_MM],
                volume_shape,
                self.config.target_spacing,
                self.config.bbox_padding_factor
            )

            if bbox is None:
                return None

            if target_size is not None:
                original_size = (volume_shape[1], volume_shape[2])

                if crop_info is not None:
                    scale, crop_offset = crop_info
                    bbox = BoundingBoxConverter.adjust_bbox_for_center_crop(
                        bbox, original_size, target_size, scale, crop_offset
                    )
                    if bbox is None:
                        return None
                else:
                    bbox = BoundingBoxConverter.adjust_bbox_for_resize(
                        bbox, original_size, target_size, preserve_aspect_ratio=True
                    )

            filename = f"{patient_id}_n{nodule_idx:02d}_z{slice_idx:04d}"
            image_dir = self.directories[f'{split}_images']
            label_dir = self.directories[f'{split}_labels']

            save_result = atomic_save_image_and_label(
                image_25d, bbox, self.config.class_id,
                image_dir / f"{filename}.jpg",
                label_dir / f"{filename}.txt"
            )

            if not save_result.success:
                return None

            metadata_entry = {
                RegModelConstants.FILE_NAME: filename,
                RegModelConstants.PATIENT_ID: patient_id,
                RegModelConstants.SPLIT_GROUP: split,
                RegModelConstants.NOUDLE_INDEX: nodule_idx,
                RegModelConstants.SLICE_INDEX: slice_idx,
                RegModelConstants.Features.FEATURE_DIAMETER_MM: features[RegModelConstants.Features.FEATURE_DIAMETER_MM],
                RegModelConstants.Features.FEATURE_MALIGNANCY: features[RegModelConstants.Features.FEATURE_MALIGNANCY],
                RegModelConstants.Features.FEATURE_SPICULATION: features[RegModelConstants.Features.FEATURE_SPICULATION],
                RegModelConstants.Features.FEATURE_LOBULATION: features[RegModelConstants.Features.FEATURE_LOBULATION],
                RegModelConstants.Features.FEATURE_SUBTLETY: features[RegModelConstants.Features.FEATURE_SUBTLETY],
                RegModelConstants.Features.FEATURE_SPHERICITY: features[RegModelConstants.Features.FEATURE_SPHERICITY],
                RegModelConstants.Features.FEATURE_MARGIN: features[RegModelConstants.Features.FEATURE_MARGIN],
                RegModelConstants.Features.FEATURE_TEXTURE: features[RegModelConstants.Features.FEATURE_TEXTURE],
                RegModelConstants.Features.FEATURE_CALCIFICATION: features[RegModelConstants.Features.FEATURE_CALCIFICATION],
                RegModelConstants.Features.FEATURE_INTERNAL_STRUCTURE: features[RegModelConstants.Features.FEATURE_INTERNAL_STRUCTURE],
                RegModelConstants.Features.FEATURE_ANNOTATION_COUNT: features[RegModelConstants.Features.FEATURE_ANNOTATION_COUNT],
                RegModelConstants.CENTROID.CENTROID_Z: centroid[0],
                RegModelConstants.CENTROID.CENTROID_Y: centroid[1],
                RegModelConstants.CENTROID.CENTROID_X: centroid[2],
                RegModelConstants.BBOX.BBOX_X: bbox[0],
                RegModelConstants.BBOX.BBOX_Y: bbox[1],
                RegModelConstants.BBOX.BBOX_W: bbox[2],
                RegModelConstants.BBOX.BBOX_H: bbox[3],
                RegModelConstants.IMAGE_PATH: save_result.image_path,
                RegModelConstants.LABEL_PATH: save_result.label_path,
                RegModelConstants.VOLUME.VOLUME_DEPTH: volume_shape[0],
                RegModelConstants.VOLUME.VOLUME_HEIGHT: volume_shape[1],
                RegModelConstants.VOLUME.VOLUME_WIDTH: volume_shape[2]
            }
            return metadata_entry

        except Exception as e:
            self.logger.error(f"[{patient_id}] Slice processing error: {e}")
            self.logger.debug(traceback.format_exc())
            return None

    def process_nodule(self, nodule_idx, annotations, features, patient_id, split, volume, volume_shape, original_spacing):
        nodule_results = []
        try:
            centroid = NoduleAnnotationProcessor.get_nodule_centroid(
                annotations, volume_shape, original_spacing, self.config.target_spacing
            )
            if centroid is None: return []

            slice_indices = NoduleAnnotationProcessor.get_nodule_slice_indices(
                annotations, volume_shape[0], original_spacing, self.config.target_spacing
            )

            selected_slices = self._select_representative_slices(
                slice_indices, self.config.slices_per_nodule
            )

            for slice_idx in selected_slices:
                metadata = self.process_single_slice(
                    slice_idx, nodule_idx, centroid, features, volume, volume_shape, patient_id, split
                )
                if metadata: nodule_results.append(metadata)

        except Exception as e:
            self.logger.error(f"[{patient_id}] Nodule error: {e}")

        return nodule_results

    def process_scan(self, scan, split, pl_module=None):
        metadata_rows = []
        patient_id = scan.patient_id

        volume_result = self.prepare_volume(scan, patient_id)
        if volume_result is None: return []

        volume, volume_shape, original_spacing = volume_result

        valid_nodules = self.extract_valid_nodules(scan, patient_id)

        for idx, (annotations, features) in enumerate(valid_nodules):
            rows = self.process_nodule(
                idx, annotations, features, patient_id, split, volume, volume_shape, original_spacing
            )
            metadata_rows.extend(rows)

        return metadata_rows

    @staticmethod
    def _select_representative_slices(slice_indices: List[int], num_slices: int) -> List[int]:
        total = len(slice_indices)
        if total == 0: return []
        if total <= num_slices: return slice_indices
        if num_slices > 1:
            return [slice_indices[int(i * (total - 1) / (num_slices - 1))] for i in range(num_slices)]
        return [slice_indices[total // 2]]
