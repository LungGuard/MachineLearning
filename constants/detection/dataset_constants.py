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
    MARGIN_FACTOR = 0.1
    MIN_CROP_SIZE = 4
    DEFAULT_CROP_SIZE = 64

