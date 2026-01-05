from enum import StrEnum
from pathlib import Path

class HuggingFaceDataSetFields(StrEnum):
    CACHE_DIR = "../datasets/hugging_face_dataset"
    DATASET_NAME = "dorsar/lung-cancer"

class DatasetConstants:
    TRAIN_SPLIT_NAME = "train" 
    TEST_SPLIT_NAME = "test" 
    VAL_SPLIT_NAME = "validation" 
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATASETS_DIR = PROJECT_ROOT / "ClassificationModel" / "datasets"
    FIGSHARE_DIR = DATASETS_DIR / "figshare_dataset"
    HUGGINGFACE_CACHE = DATASETS_DIR / "hugging_face_dataset"
    UNIFIED_DATASET_NAME = "unified_dataset"
    UNIFIED_DATASET_DIR = DATASETS_DIR / UNIFIED_DATASET_NAME
    UNIFIED_DATASET_TEST_DIR = UNIFIED_DATASET_DIR / TEST_SPLIT_NAME
    UNIFIED_DATASET_TRAIN_DIR = UNIFIED_DATASET_DIR / TRAIN_SPLIT_NAME
    UNIFIED_DATASET_VAL_DIR = UNIFIED_DATASET_DIR / VAL_SPLIT_NAME
    IMAGE_SIZE=(224, 224)
    CHANNELS=1
    BATCH_SIZE = 32
    DATASET_LABEL_MODE='categorical'
    SEED = 42
    CLASS_NAMES_KEY = 'class_names'
    NUM_CLASSES_KEY = 'num_classes'

