from enum import StrEnum


class HuggingFaceDataSetFields(StrEnum):
    CACHE_DIR = "../datasets/hugging_face_dataset"
    DATASET_NAME = "dorsar/lung-cancer"