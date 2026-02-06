"""
CT Scan Processing Module - MONAI Powered (Corrected)

Uses MONAI transforms for medical image preprocessing with proper:
- Offset detection for uncalibrated scans
- Lung windowing (center=-600, width=1500)
- Resize to fixed 512x512 output
"""

import logging
import traceback
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

import torch
from monai.transforms import ScaleIntensityRange, Zoom

from constants.detection.dataset_constants import RegModelConstants

import sys
from pathlib import Path as PathLib
sys.path.insert(0, str(PathLib(__file__).parent.parent))

from ..utils import (
    VolumePreprocessor,
    CoordinateTransformer,
    BoundingBoxConverter,
    NoduleAnnotationProcessor
)

from .file_io import atomic_save_image_and_label

logger = logging.getLogger(__name__)


class CTScanProcessor:
    """Processor for CT scans - handles volume preparation using MONAI Classes."""
    
    def __init__(self, config, directories: Dict[str, Path]):
        self.config = config
        self.directories = directories
        self.logger = logging.getLogger(__name__)
    
    def _load_volume(self, scan, patient_id: str):
        """Load volume from scan using pylidc."""
        try:
            volume = scan.to_volume()
            return volume
        except Exception as e:
            self.logger.error(f"[{patient_id}] FAILED at volume loading: {e}")
            return None
    
    def _get_spacing(self, scan, patient_id: str):
        """Extract spacing information."""
        try:
            spacing_val = scan.pixel_spacing
            if isinstance(spacing_val, (float, int, np.floating, np.integer)):
                 xy_spacing = [float(spacing_val), float(spacing_val)]
            else:
                 xy_spacing = [float(spacing_val[0]), float(spacing_val[1])]
            
            slice_spacing = float(scan.slice_spacing)
            # MONAI Order: (Z, Y, X)
            original_spacing = (slice_spacing, xy_spacing[0], xy_spacing[1])
            return original_spacing
        except Exception as e:
            self.logger.error(f"[{patient_id}] FAILED at spacing extraction: {e}")
            return None
    
    def prepare_volume(self, scan, patient_id: str):
        """
        Load, Clean (with offset detection), Resample, and Window using MONAI transforms.

        Pipeline order:
        1. Load raw volume
        2. Clean with offset detection (numpy) - handles uncalibrated scans
        3. Resample to isotropic spacing (MONAI Zoom)
        4. Apply lung windowing (MONAI ScaleIntensityRange)
        """
        # 1. Load Raw Data
        volume_np = self._load_volume(scan, patient_id)
        if volume_np is None:
            return None

        original_spacing = self._get_spacing(scan, patient_id)
        if original_spacing is None:
            return None

        try:
            # 2. CLEAN WITH OFFSET DETECTION (numpy - before any transforms)
            # This handles: NaNs, padding values, and offset scans
            volume_np = self._clean_volume_with_offset_detection(volume_np, patient_id)

            # 3. Convert to PyTorch Tensor (Channel First: 1, D, H, W)
            volume_t = torch.from_numpy(volume_np).float().unsqueeze(0)

            # 4. CRITICAL: Clamp intensity BEFORE interpolation
            # This ensures padding values don't get interpolated during Zoom
            # Equivalent to MONAI's ClampIntensity but without import issues
            volume_t = torch.clamp(volume_t, min=-1000.0, max=3000.0)

            # 5. SPATIAL RESAMPLING using MONAI Zoom
            zoom_factors = [
                orig / target
                for orig, target in zip(original_spacing, self.config.target_spacing)
            ]

            zoomer = Zoom(
                zoom=zoom_factors,
                mode="bilinear",
                padding_mode="border",  # Prevents grey border artifacts
                keep_size=False
            )
            vol_resampled = zoomer(volume_t)

            # 6. LUNG WINDOWING using MONAI ScaleIntensityRange
            # Lung window: center=-600, width=1500 -> range [-1350, +150]
            window_center = getattr(self.config, 'window_center', -600.0)
            window_width = getattr(self.config, 'window_width', 1500.0)
            window_min = window_center - (window_width / 2.0)  # -1350
            window_max = window_center + (window_width / 2.0)  # +150

            scaler = ScaleIntensityRange(
                a_min=window_min,
                a_max=window_max,
                b_min=0.0,
                b_max=1.0,
                clip=True
            )
            vol_windowed = scaler(vol_resampled)

            # 7. Convert back to Numpy
            final_volume = vol_windowed[0].numpy()

            self.logger.debug(
                f"[{patient_id}] MONAI processing complete: shape={final_volume.shape}, "
                f"range=[{final_volume.min():.3f}, {final_volume.max():.3f}]"
            )

            return final_volume, final_volume.shape, original_spacing

        except Exception as e:
            self.logger.error(f"[{patient_id}] MONAI Processing FAILED: {e}")
            self.logger.debug(traceback.format_exc())
            return None

    def _clean_volume_with_offset_detection(self, volume: np.ndarray, patient_id: str) -> np.ndarray:
        """
        Clean volume with offset detection - handles uncalibrated LIDC scans.

        Operations:
        1. Handle NaNs
        2. Detect and handle padding values (< -1500)
        3. Detect offset scans (5th percentile > -100) and apply -1024 correction
        4. Set padding to air (-1000 HU)
        5. Clip to valid HU range
        """
        volume = volume.astype(np.float32)

        # Handle NaNs
        nan_count = np.isnan(volume).sum()
        if nan_count > 0:
            self.logger.warning(f"[{patient_id}] Found {nan_count} NaN values, replacing with -1000")
            volume = np.nan_to_num(volume, nan=-1000.0)

        # Create padding mask (values < -1500 are padding: -2048, -3024, etc.)
        PADDING_THRESHOLD = -1500.0
        padding_mask = volume < PADDING_THRESHOLD
        padding_count = padding_mask.sum()

        # Detect offset using center slice
        depth = volume.shape[0]
        center_slice = volume[depth // 2]
        valid_mask = (center_slice > PADDING_THRESHOLD) & (center_slice < 4000)
        valid_pixels = center_slice[valid_mask]

        offset_applied = False
        if len(valid_pixels) > 0:
            percentile_5 = np.percentile(valid_pixels, 5)
            OFFSET_THRESHOLD = -100.0

            if percentile_5 > OFFSET_THRESHOLD:
                self.logger.info(
                    f"[{patient_id}] Offset scan detected (P5={percentile_5:.1f}). "
                    f"Applying -1024 correction."
                )
                volume[~padding_mask] -= 1024.0
                offset_applied = True

        # Set padding to air BEFORE resampling
        if padding_count > 0:
            volume[padding_mask] = -1000.0

        # Clip to valid HU range
        volume = np.clip(volume, -1000.0, 3000.0)

        self.logger.debug(
            f"[{patient_id}] Volume cleaned: min={volume.min():.1f}, max={volume.max():.1f}, "
            f"offset_applied={offset_applied}"
        )

        return volume

    def extract_valid_nodules(self, scan, patient_id: str) -> List[Tuple]:
        """Extract and filter valid nodules from scan."""
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

    def process_single_slice(self, slice_idx, nodule_idx, centroid, features, windowed, volume_shape, patient_id, split):
        """Process single slice: create 2.5D image, compute bbox, center crop, save, generate metadata."""
        try:
            # Get target size from config
            target_size = getattr(self.config, 'output_image_size', (512, 512))
            use_center_crop = getattr(self.config, 'use_center_crop', True)
            max_scale = getattr(self.config, 'max_scale', 1.2)

            # Create 2.5D image with center crop (returns image and crop_info)
            image_25d, crop_info = VolumePreprocessor.create_25d_sandwich(
                windowed,
                slice_idx,
                target_size=target_size,
                use_center_crop=use_center_crop,
                max_scale=max_scale
            )

            if image_25d is None or image_25d.size == 0:
                return None

            # Compute bbox in original volume coordinates
            bbox = BoundingBoxConverter.compute_nodule_bbox_yolo(
                centroid,
                features[RegModelConstants.Features.FEATURE_DIAMETER_MM],
                volume_shape,
                self.config.target_spacing,
                self.config.bbox_padding_factor
            )

            if bbox is None:
                return None

            # Adjust bbox for center crop or padding
            if target_size is not None:
                original_size = (volume_shape[1], volume_shape[2])  # (height, width)

                if crop_info is not None:
                    # Center crop was used - adjust bbox accordingly
                    scale, crop_offset = crop_info
                    bbox = BoundingBoxConverter.adjust_bbox_for_center_crop(
                        bbox,
                        original_size,
                        target_size,
                        scale,
                        crop_offset
                    )
                    # If bbox is None, nodule was cropped out of frame
                    if bbox is None:
                        self.logger.debug(
                            f"[{patient_id}] Slice {slice_idx}: nodule center cropped out of frame"
                        )
                        return None
                else:
                    # Padding was used
                    bbox = BoundingBoxConverter.adjust_bbox_for_resize(
                        bbox,
                        original_size,
                        target_size,
                        preserve_aspect_ratio=True
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

    def process_nodule(self, nodule_idx, annotations, features, patient_id, split, windowed, volume_shape, original_spacing):
        """Process single nodule and generate metadata for its slices."""
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
                meta = self.process_single_slice(
                    slice_idx, nodule_idx, centroid, features, windowed, volume_shape, patient_id, split
                )
                if meta: nodule_results.append(meta)
                
        except Exception as e:
            self.logger.error(f"[{patient_id}] Nodule error: {e}")
            
        return nodule_results

    def process_scan(self, scan, split, pl_module=None):
        """Process single CT scan (main orchestrator)."""
        metadata_rows = []
        patient_id = scan.patient_id
        
        volume_result = self.prepare_volume(scan, patient_id)
        if volume_result is None: return []
        
        windowed, volume_shape, original_spacing = volume_result
        
        valid_nodules = self.extract_valid_nodules(scan, patient_id)
        
        for idx, (anns, feats) in enumerate(valid_nodules):
            rows = self.process_nodule(
                idx, anns, feats, patient_id, split, windowed, volume_shape, original_spacing
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