from pathlib import Path

PROJECT_ROOT_DIR = Path(__file__).parents[3]

CHECKPOINT_DIR_PATH = PROJECT_ROOT_DIR / "DetectionModel" / "src" / "model_checkpoints"

LOG_DIR_PATH = PROJECT_ROOT_DIR / "logs"


class DetectionModelConstants:
    PROJECT_ROOT = PROJECT_ROOT_DIR
    DATASET_DIR = PROJECT_ROOT / "datasets"
    MODEL_NAME = "Detection Model"
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 16
    DATASET_YAML = DATASET_DIR / "data.yaml"
    LOG_DIR = LOG_DIR_PATH
