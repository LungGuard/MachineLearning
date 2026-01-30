from pathlib import Path


class RegModelConstants:
    

    DATASET_METADATA_PATH = Path(__file__).parent.parent.parent / "DetectionModel" / "datasets" / "metadata" / "regression_dataset.csv"
    
    FILE_NAME = 'filename'
    PATIENT_ID = 'patient_id'
    SPLIT_GROUP = 'split_group'
    NOUDLE_INDEX = 'nodule_index'
    SLICE_INDEX = 'slice_index'
    IMAGE_PATH = 'image_path'
    LABEL_PATH = 'label_path'

    TRAIN_SPLIT="train"
    TEST_SPLIT="test"
    VAL_SPLIT = "val"

    DATASET_X_FEATURES= "x_features"
    DATASET_Y_FEATURES= "y_features"
    
    class Features:
        FEATURE_DIAMETER_MM = 'diameter_mm'
        FEATURE_MALIGNANCY = 'malignancy'
        FEATURE_SPICULATION = 'spiculation'
        FEATURE_LOBULATION = 'lobulation'
        FEATURE_SUBTLETY = 'subtlety'
        FEATURE_SPHERICITY = 'sphericity'
        FEATURE_MARGIN='margin'
        FEATURE_TEXTURE = 'texture'
        FEATURE_CALCIFICATION = 'calcification'
        FEATURE_ANNOTATION_COUNT = 'annotation_count'
        FEATURE_INTERNAL_STRUCTURE = 'internal_structure'
        DEFAULT_FEATURES = {
            FEATURE_MALIGNANCY : 3.0,  # Indeterminate
            FEATURE_SPICULATION : 1.0,
            FEATURE_LOBULATION : 1.0,
            FEATURE_SUBTLETY : 3.0,
            FEATURE_SPHERICITY : 3.0,
            FEATURE_MARGIN : 3.0,
            FEATURE_TEXTURE : 3.0,
            FEATURE_CALCIFICATION : 1.0,
            FEATURE_INTERNAL_STRUCTURE : 1.0,
        }
    class CENTROID:
        CENTROID_X = 'centroid_x'
        CENTROID_Y = 'centroid_Y'
        CENTROID_Z = 'centroid_Z'
        CENTROID_DIM = 3
    class BBOX:
        BBOX_X = 'bbox_x'
        BBOX_Y = 'bbox_y'
        BBOX_W = 'bbox_w'
        BBOX_H = 'bbox_h'
    class VOLUME :
        VOLUME_DEPTH = 'volume_depth'
        VOLUME_HEIGHT = 'volume_height'
        VOLUME_WIDTH = 'volume_width'