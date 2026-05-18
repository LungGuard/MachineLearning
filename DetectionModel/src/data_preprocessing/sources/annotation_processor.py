"""
Nodule Annotation Processor Module
Utilities for processing nodule annotations from radiologists.
"""

import logging
from itertools import chain
from typing import Tuple, Optional, List

import numpy as np

from DetectionModel.constants.enums.features import Features, DEFAULT_FEATURES

logger = logging.getLogger(__name__)

# Maps Features enum members to their corresponding pylidc annotation attribute names.
_FEATURE_ATTR: dict = {
    Features.MALIGNANCY:         "malignancy",
    Features.SPICULATION:        "spiculation",
    Features.LOBULATION:         "lobulation",
    Features.SUBTLETY:           "subtlety",
    Features.SPHERICITY:         "sphericity",
    Features.MARGIN:             "margin",
    Features.TEXTURE:            "texture",
    Features.CALCIFICATION:      "calcification",
    Features.INTERNAL_STRUCTURE: "internalStructure",
}


class NoduleAnnotationProcessor:
    """Utilities for processing nodule annotations from radiologists."""

    @staticmethod
    def _safe_extract_bbox(ann):
        """Safely extract bounding box from annotation."""
        try:
            bbox = ann.bbox()
            return bbox
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(f"bbox extraction failed: {e}")
            return None

    @staticmethod
    def _safe_extract_centroid(ann):
        """Safely extract centroid from annotation."""
        try:
            centroid = ann.centroid
            return centroid if centroid and len(centroid) == 3 else None
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(f"centroid extraction failed: {e}")
            return None

    @staticmethod
    def _safe_extract_contour_slice_indices(ann):
        """Safely extract contour slice indices from annotation."""
        try:
            return ann.contour_slice_indices
        except Exception:
            return None

    @staticmethod
    def extract_nodule_features(
        annotations: list,
        fallback_diameter: float = 10.0
    ) -> dict:
        """
        Extract and aggregate features from multiple radiologist annotations.

        LIDC-IDRI nodules have 1-4 independent radiologist annotations.
        This function computes consensus features using averaging.
        """
        # Safe copy — avoids mutating the module-level DEFAULT_FEATURES constant.
        default_features = {**DEFAULT_FEATURES, Features.DIAMETER_MM: fallback_diameter}

        def get_feature_value(feature_scores, feature_key):
            """Return the feature value if scores exist, otherwise return default."""
            return (float(np.mean(feature_scores)) if feature_scores
                    else default_features[feature_key])

        has_annotations = len(annotations) > 0
        scores = (
            {feat: [getattr(ann, attr) for ann in annotations]
             for feat, attr in _FEATURE_ATTR.items()}
            if has_annotations
            else {feat: [] for feat in _FEATURE_ATTR}
        )
        diameters = [float(ann.diameter) for ann in annotations] if has_annotations else []

        return {
            Features.DIAMETER_MM: get_feature_value(diameters, Features.DIAMETER_MM),
            **{feat: get_feature_value(scores[feat], feat) for feat in _FEATURE_ATTR},
            Features.ANNOTATION_COUNT: len(annotations),
        }

    @staticmethod
    def _get_valid_slice_indices_from_annotation(
        ann,
        z_scale: float,
        volume_depth: int
    ) -> List[int]:
        """Extract and transform valid slice indices from a single annotation."""
        try:
            original_indices = ann.contour_slice_indices
            transformed_indices = [int(round(idx * z_scale)) for idx in original_indices]
            valid_indices = [idx for idx in transformed_indices if 0 <= idx < volume_depth]
            return valid_indices
        except Exception:
            return []

    @staticmethod
    def get_nodule_slice_indices(
        annotations: List,
        volume_depth: int,
        original_spacing: Tuple[float, float, float] = None,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> List[int]:
        """Get valid slice indices for a nodule, transformed to resampled space."""
        z_scale = (
            original_spacing[0] / target_spacing[0]
            if original_spacing is not None
            else 1.0
        )

        unique_indices = set(chain.from_iterable(
            NoduleAnnotationProcessor._get_valid_slice_indices_from_annotation(
                ann, z_scale, volume_depth
            )
            for ann in annotations
        ))

        return sorted(unique_indices)

    @staticmethod
    def get_nodule_centroid(
        annotations: List,
        volume_shape: Tuple[int, int, int],
        original_spacing: Tuple[float, float, float] = None,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> Optional[Tuple[float, float, float]]:
        """Calculate nodule centroid from annotations, transformed to resampled space."""
        centroids = []

        for ann in annotations:
            try:
                centroid = ann.centroid  # pylidc returns (i, j, k) = (row, col, slice)
                if centroid is not None and len(centroid) == 3:
                    # Reorder to (slice, row, col) = (z, y, x) to match the transposed volume.
                    centroids.append((centroid[2], centroid[0], centroid[1]))
            except (AttributeError, TypeError, ValueError) as e:
                logger.warning(f"Annotation centroid extraction failed: {e}")

        if not centroids:
            return None

        og_space_avg_centroid = tuple(np.mean(centroids, axis=0))

        # Transform to resampled space if spacing provided (inline CoordinateTransformer math)
        if original_spacing:
            scale_factors = tuple(
                orig / tgt for orig, tgt in zip(original_spacing, target_spacing)
            )
            transformed_centroid = tuple(
                coord * scale for coord, scale in zip(og_space_avg_centroid, scale_factors)
            )
        else:
            transformed_centroid = og_space_avg_centroid

        # Validate against resampled volume bounds
        z, y, x = transformed_centroid
        is_within_bounds = (
            0 <= z < volume_shape[0] and
            0 <= y < volume_shape[1] and
            0 <= x < volume_shape[2]
        )

        return transformed_centroid if is_within_bounds else None
