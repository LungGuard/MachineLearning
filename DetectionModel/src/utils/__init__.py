"""
Detection Model Utilities Package

Re-exports from canonical locations in data_preprocessing package.
These classes were moved into data_preprocessing for better organization.
"""

# Re-export from new canonical locations
from ..data_preprocessing.preprocessing.slice_processor import SlicePreprocessor as VolumePreprocessor
from ..data_preprocessing.core.coordinate_transformer import CoordinateTransformer
from ..data_preprocessing.preprocessing.bbox_converter import BoundingBoxConverter
from ..data_preprocessing.sources.annotation_processor import NoduleAnnotationProcessor

# Export all classes
__all__ = [
    'VolumePreprocessor',
    'CoordinateTransformer',
    'BoundingBoxConverter',
    'NoduleAnnotationProcessor',
]


resample_volume = VolumePreprocessor.resample_volume
apply_windowing = VolumePreprocessor.apply_windowing
create_25d_sandwich = VolumePreprocessor.create_25d_sandwich

transform_coordinates_to_resampled = CoordinateTransformer.transform_coordinates_to_resampled
transform_slice_to_resampled_space = CoordinateTransformer.transform_slice_to_resampled_space
is_slice_within_volume = CoordinateTransformer.is_slice_within_volume

compute_nodule_bbox_yolo = BoundingBoxConverter.compute_nodule_bbox_yolo
convert_to_yolo_format = BoundingBoxConverter.convert_to_yolo_format
compute_diameter = BoundingBoxConverter.compute_diameter

extract_nodule_features = NoduleAnnotationProcessor.extract_nodule_features
get_nodule_slice_indices = NoduleAnnotationProcessor.get_nodule_slice_indices
get_nodule_centroid = NoduleAnnotationProcessor.get_nodule_centroid
