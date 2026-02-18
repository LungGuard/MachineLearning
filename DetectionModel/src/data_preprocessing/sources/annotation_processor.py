"""
Nodule Annotation Processor Module
Utilities for processing nodule annotations from radiologists.
"""

import numpy as np
from typing import Tuple, Optional, List
from constants.detection.dataset_constants import DatasetConstants
from constants.detection.centorid_enum import CENTROID
from constants.detection.features_enum import Features
from ..core.coordinate_transformer import CoordinateTransformer
from ..preprocessing.bbox_converter import BoundingBoxConverter
import contextlib


class NoduleAnnotationProcessor:
    """Utilities for processing nodule annotations from radiologists."""
    
    import contextlib
    @staticmethod
    def _safe_extract_bbox(ann):
        """Safely extract bounding box from annotation."""
        try:            
            bbox = ann.bbox()
            return bbox
        except Exception:
            return None
    
    @staticmethod
    def _safe_extract_centroid(ann):
        """Safely extract centroid from annotation."""
        try:
            centroid = ann.centroid
            # Validate centroid exists and has correct dimensions
            return centroid if centroid is not None and len(centroid) == 3 else None
        except Exception:
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
        default_features = DatasetConstants.Features.DEFAULT_FEATURES
        default_features[DatasetConstants.Features.FEATURE_DIAMETER_MM] = fallback_diameter

        def get_feature_value(feature_scores, feature_key):
            """Return the feature value if scores exist, otherwise return default."""
            return (float(np.mean(feature_scores)) if feature_scores 
                    else default_features[feature_key])    
        
        has_annotations = len(annotations) > 0
        
        malignancy_scores = [ann.malignancy for ann in annotations] if has_annotations else []
        spiculation_scores = [ann.spiculation for ann in annotations] if has_annotations else []
        lobulation_scores = [ann.lobulation for ann in annotations] if has_annotations else []
        subtlety_scores = [ann.subtlety for ann in annotations] if has_annotations else []
        sphericity_scores = [ann.sphericity for ann in annotations] if has_annotations else []
        margin_scores = [ann.margin for ann in annotations] if has_annotations else []
        texture_scores = [ann.texture for ann in annotations] if has_annotations else []
        calcification_scores = [ann.calcification for ann in annotations] if has_annotations else []
        internal_structure_scores = [ann.internalStructure for ann in annotations] if has_annotations else []
        
        bboxs = filter(None, map(NoduleAnnotationProcessor._safe_extract_bbox, annotations))
        diameters = list(map(BoundingBoxConverter.compute_diameter, bboxs))
        
        return  {
            Features.DIAMETER_MM.value: get_feature_value(
                diameters, Features.DIAMETER_MM.value),
            Features.MALIGNANCY.value: get_feature_value(
                malignancy_scores,Features.MALIGNANCY.value),
            Features.SPICULATION.value: get_feature_value(
                spiculation_scores, Features.SPICULATION.value),
            Features.LOBULATION.value: get_feature_value(
                lobulation_scores, Features.LOBULATION.value),
            Features.SUBTLETY.value: get_feature_value(
                subtlety_scores,Features.SUBTLETY.value),
            Features.SPHERICITY.value: get_feature_value(
                sphericity_scores, Features.SPHERICITY.value),
            Features.MARGIN.value: get_feature_value(
                margin_scores, Features.MARGIN.value),
            Features.TEXTURE.value: get_feature_value(
                texture_scores,Features.TEXTURE.value),
            Features.CALCIFICATION.value: get_feature_value(
                calcification_scores, Features.CALCIFICATION.value),
            Features.INTERNAL_STRUCTURE.value : get_feature_value(
                internal_structure_scores,Features.INTERNAL_STRUCTURE.value),
            Features.ANNOTATION_COUNT.value: len(annotations)
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
            transformed_indices = map(
                lambda idx: CoordinateTransformer.transform_slice_to_resampled_space(idx, z_scale),
                original_indices
            )
            valid_indices = filter(
                lambda idx: CoordinateTransformer.is_slice_within_volume(idx, volume_depth),
                transformed_indices
            )
            return list(valid_indices)
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

        all_valid_indices = map(
            lambda ann: NoduleAnnotationProcessor._get_valid_slice_indices_from_annotation(
                ann, z_scale, volume_depth
            ),
            annotations
        )

        unique_indices = {idx for indices in all_valid_indices for idx in indices}

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
            with contextlib.suppress(Exception):
                centroid = ann.centroid  # Returns (z, y, x) in original space
                if centroid is not None and len(centroid) == 3:
                    centroids.append(centroid)
        if not centroids:
            return None

        # Average centroid in original space
        avg_centroid = tuple(
            sum(c[i] for c in centroids) / len(centroids)
            for i in range(len(CENTROID))
        )

        # Transform to resampled space if spacing provided
        transformed_centroid = (
            CoordinateTransformer.transform_coordinates_to_resampled(
                avg_centroid, original_spacing, target_spacing
            )
            if original_spacing is not None
            else avg_centroid
        )

        # Validate against resampled volume bounds
        z, y, x = transformed_centroid
        is_within_bounds = (
            0 <= z < volume_shape[0] and
            0 <= y < volume_shape[1] and
            0 <= x < volume_shape[2]
        )

        return transformed_centroid if is_within_bounds else None
