from enum import StrEnum
from paths import ProjectPaths


class HuggingFaceDataSetFields:
    CACHE_DIR = ProjectPaths.CLASSIFICATION_HUGGINGFACE_CACHE
    DATASET_NAME = "dorsar/lung-cancer"
