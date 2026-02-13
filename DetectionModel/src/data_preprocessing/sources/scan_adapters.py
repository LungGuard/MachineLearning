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


# ──────────────────────────────────────────────
# Adapter 1: PyLIDC (Training / Data Preparation)
# ──────────────────────────────────────────────

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


# ──────────────────────────────────────────────
# Adapter 2: Raw DICOM via MONAI (Inference)
# ──────────────────────────────────────────────

class DICOMScanSource:
    """Loads a CT scan from a DICOM directory using MONAI."""

    def __init__(self, dicom_dir: Path, patient_id_override: str = None):
        self._dicom_dir = Path(dicom_dir)
        self._patient_id = patient_id_override or self._dicom_dir.name

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
            pixdim = meta.get('pixdim', None)
            affine = meta.get('affine', None)
            spacing = (
                (float(pixdim[3]), float(pixdim[1]), float(pixdim[2]))
                if pixdim is not None and len(pixdim) >= 4
                else (
                    tuple(map(lambda i: float(np.abs(affine[i, i])), (2, 0, 1)))
                    if affine is not None
                    else None
                )
            )
        except Exception as e:
            logger.error(f"Spacing extraction from DICOM meta failed: {e}")
            spacing = None
        return spacing