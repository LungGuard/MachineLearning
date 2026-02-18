from enum import Enum,StrEnum
class DatasetConstants:
    
    FILE_NAME = 'filename'
    PATIENT_ID = 'patient_id'
    SPLIT_GROUP = 'split_group'
    NOUDLE_INDEX = 'nodule_index'
    SLICE_INDEX = 'slice_index'
    IMAGE_PATH = 'image_path'
    LABEL_PATH = 'label_path'


    TRAIN_IMAGES='train_images'
    TRAIN_LABELS='train_labels'
    VAL_IMAGES='val_images'
    VAL_LABELS='val_labels'
    TEST_IMAGES = 'test_images'
    TEST_LABELS = 'test_labels'
    METADATA = 'metadata'

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

    
class PreProcessingConstants:
    class HU_VALUES:
        AIR_HU = -1000
        MAX_HU = 3000
        PADDING_THRESHOLD = -1500
        OFFSET_THRESHOLD = -100
        OFFSET_CORRECTION = 1024
        VALID_PIXEL_MAX = 4000
        OFFSET_PERCENTILE = 5

    

    TARGET_SPACING = (1.0, 1.0, 1.0)
    WINDOW_CENTER = -600.0
    WINDOW_WIDTH = 1500.0

    MAX_CROP_SCALE = 1.5

    class INTENSITY_RANGE:
        OUTPUT_MIN = 0.0
        OUTPUT_MAX = 1.0

class Features(StrEnum):
        FEATURE_DIAMETER_MM = 'diameter_mm'
        FEATURE_MALIGNANCY = 'malignancy'
        FEATURE_SPICULATION = 'spiculation'
        FEATURE_LOBULATION = 'lobulation'
        FEATURE_SUBTLETY = 'subtlety'
        FEATURE_SPHERICITY = 'sphericity'
        FEATURE_MARGIN='margin'
        FEATURE_TEXTURE = 'texture'
        FEATURE_CALCIFICATION = 'calcification'
        FEATURE_INTERNAL_STRUCTURE = 'internal_structure'