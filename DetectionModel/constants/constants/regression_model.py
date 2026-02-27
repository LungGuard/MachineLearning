from paths import ProjectPaths

CHECKPOINT_DIR_PATH = ProjectPaths.DETECTION_CHECKPOINT_DIR

LOG_DIR_PATH = ProjectPaths.LOGS_DIR


class RegressionModelConstants:
    MODEL_NAME = "Nodule Features Prediction Model"
    PROJECT_ROOT = ProjectPaths.ROOT
    DEFAULT_INPUT_SHAPE = (3, 64, 64)
    DEFAULT_LEARNING_RATE = 1e-4
    CONV_BLOCK_NAME_PREFIX = 'conv'
    DENSE_BLOCK_NAME_PREFIX = 'dense'
    BRIDGE_LAYER_NAME = 'bridge'
    FLATTEN_LAYER_NAME = 'flatten'
    OUTPUT_LAYER_NAME = 'output'
    CHECKPOINT_DIR = ProjectPaths.DETECTION_CHECKPOINT_DIR
    EXCEPTION_CHECKPOINT_FILE_NAME = "exception_reg_model_checkpoint"
    BEST_MODEL_CHECKPOINT_NAME = "best_reg_model"
    LOG_DIR = ProjectPaths.LOGS_DIR
    
