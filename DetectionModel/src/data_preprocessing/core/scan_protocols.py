"""Scan Source Protocol & Data Containers.

Defines:
  1. ScanSource protocol — the contract for any scan source
  2. Data containers for BOTH pipelines:
     • VolumeData, NoduleData           → shared / data-prep
     • YOLODetection, ProcessedSlice,
       NoduleCropResult                 → inference
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, List, Tuple, Dict, Optional, runtime_checkable

import numpy as np


# ──────────────────────────────────────────────
# Shared Data Containers
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class VolumeData:
    """Immutable container for a loaded CT volume and its spatial metadata."""
    volume: np.ndarray
    spacing: Tuple[float, float, float]


@dataclass(frozen=True)
class NoduleData:
    """Source-agnostic representation of a single nodule (data-prep)."""
    index: int
    centroid_zyx: Tuple[float, float, float]
    features: Dict[str, float]
    slice_indices: List[int]
    raw_annotations: object = field(default=None, repr=False)


# ──────────────────────────────────────────────
# Inference Data Containers
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class YOLODetection:
    """Single YOLO detection on one axial slice."""
    slice_index: int
    bbox_xywh_norm: Tuple[float, float, float, float]
    confidence: float
    class_id: int = 0


@dataclass(frozen=True)
class ProcessedSlice:
    """In-memory result of processing one volume slice for inference."""
    slice_index: int
    enhanced_25d: np.ndarray
    middle_slice: np.ndarray
    quality_passed: bool
    reject_reason: str = "OK"


@dataclass(frozen=True)
class NoduleCropResult:
    """A detected nodule region ready for the Stage 2 CNN classifier."""
    detection: YOLODetection
    full_slice_enhanced: np.ndarray
    nodule_crop_single: np.ndarray
    nodule_crop_25d: np.ndarray


# ──────────────────────────────────────────────
# Protocol
# ──────────────────────────────────────────────

@runtime_checkable
class ScanSource(Protocol):
    """Contract every scan source must satisfy."""

    @property
    def patient_id(self) -> str: ...

    def load_volume(self) -> Optional[VolumeData]: ...

    def extract_nodules(self, volume_shape: Tuple[int, int, int],
                        original_spacing: Tuple[float, float, float],
                        target_spacing: Tuple[float, float, float]) -> List[NoduleData]: ...