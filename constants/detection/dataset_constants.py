
class RegModelConstants:
    
    FILE_NAME = 'filename'
    PATIENT_ID = 'patient_id'
    SPLIT_GROUP = 'split_group'
    NOUDLE_INDEX = 'nodule_index'
    SLICE_INDEX = 'slice_index'
    IMAGE_PATH = 'image_path'
    LABEL_PATH = 'label_path'

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
    class CENTROID:
        CENTROID_X = 'centroid_x'
        CENTROID_Y = 'centroid_Y'
        CENTROID_Z = 'centroid_Z'
    class BBOX:
        BBOX_X = 'bbox_x'
        BBOX_Y = 'bbox_y'
        BBOX_W = 'bbox_w'
        BBOX_H = 'bbox_h'
    class VOLUME :
        VOLUME_DEPTH = 'volume_depth'
        VOLUME_HEIGHT = 'volume_height'
        VOLUME_WIDTH = 'volume_width'