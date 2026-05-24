"""Scan Source Adapters.

PyLIDCScanSource  → wraps pylidc.Scan for data-preparation
DICOMScanSource   → wraps raw DICOM dirs via MONAI for inference
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

from ..core.scan_protocols import ScanSource, VolumeData, NoduleData

logger = logging.getLogger(__name__)



class PyLIDCScanSource:
    """Wraps a pylidc.Scan object to satisfy the ScanSource protocol."""

    def __init__(self, scan, annotation_processor_cls=None):
        self._scan = scan
        self._annotation_processor = annotation_processor_cls

    @property
    def patient_id(self) -> str:
        return self._scan.patient_id

    def load_volume(self) -> Optional[VolumeData]:
        try:
            raw_volume = self._scan.to_volume()
            # pylidc stacks slices on axis=-1: shape is (N_rows, N_cols, N_slices).
            # The pipeline expects (N_slices, N_rows, N_cols) = (z, y, x).
            raw_volume = np.transpose(raw_volume, (2, 0, 1))
            spacing = self._extract_spacing()
            result = VolumeData(volume=raw_volume, spacing=spacing) if spacing is not None else None
        except Exception as e:
            logger.error(f"[{self.patient_id}] PyLIDC volume load failed: {e}")
            result = None
        return result

    def extract_nodules(self, volume_shape: Tuple[int, int, int],
                        original_spacing: Tuple[float, float, float],
                        target_spacing: Tuple[float, float, float]) -> List[NoduleData]:
        proc = self._annotation_processor
        nodule_data_list: List[NoduleData] = []

        try:
            clusters = self._scan.cluster_annotations()
        except Exception:
            return nodule_data_list

        for idx, annotations in enumerate(clusters):
            features = proc.extract_nodule_features(annotations)
            centroid = proc.get_nodule_centroid(
                annotations, volume_shape, original_spacing, target_spacing
            )
            slice_indices = proc.get_nodule_slice_indices(
                annotations, volume_shape[0], original_spacing, target_spacing
            )
            nodule_data_list.append(NoduleData(
                index=idx,
                centroid_zyx=centroid,
                features=features,
                slice_indices=slice_indices,
                raw_annotations=annotations,
            )) if centroid is not None else None

        return nodule_data_list

    def _extract_spacing(self) -> Optional[Tuple[float, float, float]]:
        try:
            px_raw = self._scan.pixel_spacing
            xy = (
                [float(px_raw), float(px_raw)]
                if isinstance(px_raw, (float, int, np.floating, np.integer))
                else [float(px_raw[0]), float(px_raw[1])]
            )
            result = (float(self._scan.slice_spacing), xy[0], xy[1])
        except Exception as e:
            logger.error(f"[{self.patient_id}] Spacing extraction failed: {e}")
            result = None
        return result




class DICOMScanSource:
    """Loads a CT scan from a DICOM directory using MONAI.

    Args:
        is_lidc: When True, corrects MONAI's default ITKReader axis order
            from (x, y, z) to the pipeline convention (z, y, x) via
            np.transpose(volume, (2, 1, 0)).  Set this when loading LIDC
            DICOM directories for inference testing.  The root cause is
            ITKReader's default reverse_indexing=False, which applies to
            any DICOM source; the flag may be widened or made unconditional
            once non-LIDC inputs are validated.
    """

    def __init__(self, dicom_dir: Path, patient_id_override: str = None,
                 is_lidc: bool = True):
        self._dicom_dir = Path(dicom_dir)
        self._patient_id = patient_id_override or self._dicom_dir.name
        self._is_lidc = is_lidc

    @property
    def patient_id(self) -> str:
        return self._patient_id

    def load_volume(self) -> Optional[VolumeData]:
        try:
            from monai.transforms import LoadImage

            loader = LoadImage(image_only=False)
            data, meta = loader(str(self._dicom_dir))

            volume = data.numpy() if hasattr(data, 'numpy') else np.array(data)
            volume = volume[0] if volume.ndim == 4 else volume
            # MONAI ITKReader default (reverse_indexing=False) returns (x, y, z).
            # The pipeline expects (z, y, x) = (slices, rows, cols).
            if self._is_lidc and volume.ndim == 3:
                volume = np.transpose(volume, (2, 1, 0))

            spacing = self._extract_spacing_from_meta(meta)
            result = VolumeData(volume=volume, spacing=spacing) if spacing is not None else None
        except Exception as e:
            logger.error(f"[{self._patient_id}] DICOM load failed: {e}")
            result = None
        return result

    def extract_nodules(self, volume_shape, original_spacing, target_spacing) -> List[NoduleData]:
        """No ground-truth nodules in inference — YOLO handles detection."""
        return []

    @staticmethod
    def _extract_spacing_from_meta(meta: dict) -> Optional[Tuple[float, float, float]]:
        try:
            spacing = None
            # Prefer MONAI's pre-computed spacing key — MONAI returns (x, y, z),
            # reorder to (z, y, x) to match the rest of the pipeline.
            if 'spacing' in meta:
                raw = meta['spacing']
                if len(raw) >= 3:
                    candidate = (float(raw[2]), float(raw[1]), float(raw[0]))
                    if all(s > 0 for s in candidate):
                        spacing = candidate
            # Fall back to NIfTI pixdim: [qfac, x, y, z, ...]
            if spacing is None:
                pixdim = meta.get('pixdim', None)
                if pixdim is not None and len(pixdim) >= 4:
                    candidate = (float(pixdim[3]), float(pixdim[2]), float(pixdim[1]))
                    if all(s > 0 for s in candidate):
                        spacing = candidate
            # Last resort: diagonal of affine matrix — NIfTI affine diagonal is
            # (x_scale, y_scale, z_scale); reorder to (z, y, x).
            if spacing is None:
                affine = meta.get('affine', None)
                if affine is not None:
                    candidate = tuple(float(np.abs(affine[i, i])) for i in (2, 1, 0))
                    if all(s > 0 for s in candidate):
                        spacing = candidate
        except Exception as e:
            logger.error(f"Spacing extraction from DICOM meta failed: {e}")
            spacing = None
        return spacing