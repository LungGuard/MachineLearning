from paths import ProjectPaths


class ModelConstants:
    PADDING_SAME = 'same'
    EPOCHS = 100
    STATE_MODEL_WEIGHTS = 'model_weights'
    STATE_MODEL_CONFIG = 'model_config'
    CHECKPOINT_DIR_PATH = ProjectPaths.CLASSIFICATION_CHECKPOINT_DIR
    RESULTS_DIR_PATH = ProjectPaths.CLASSIFICATION_RESULTS_DIR
    CANCER_TYPE_RESULT_KEY = 'cancer_type'
    CONFIDENCE_KEY = 'confidence'
    MODEL_NAME = 'Cancer Classification Model'
    HORIZONTAL_FLIP_AUGMENTATION = "horizontal"
    CHECKPOINT_FILE_PATH = CHECKPOINT_DIR_PATH / 'best_model.keras'
