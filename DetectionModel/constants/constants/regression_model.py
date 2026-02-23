from pathlib import Path

PROJECT_ROOT_DIR = Path(__file__).parents[3]

CHECKPOINT_DIR_PATH = PROJECT_ROOT_DIR / "DetectionModel" / "src" / "model_checkpoints"

LOG_DIR_PATH = PROJECT_ROOT_DIR / "logs"


class RegressionModelConstants:
    MODEL_NAME = "Nodule Features Prediction Model"
    PROJECT_ROOT = PROJECT_ROOT_DIR
    DEFAULT_INPUT_SHAPE = (3, 64, 64)
    DEFAULT_LEARNING_RATE = 1e-4
    CONV_BLOCK_NAME_PREFIX = 'conv'
    DENSE_BLOCK_NAME_PREFIX = 'dense'
    BRIDGE_LAYER_NAME = 'bridge'
    FLATTEN_LAYER_NAME = 'flatten'
    OUTPUT_LAYER_NAME = 'output'
    CHECKPOINT_DIR = CHECKPOINT_DIR_PATH
    EXCEPTION_CHECKPOINT_FILE_NAME = "exception_reg_model_checkpoint"
    BEST_MODEL_CHECKPOINT_NAME = "best_reg_model"
    LOG_DIR = LOG_DIR_PATH
    
