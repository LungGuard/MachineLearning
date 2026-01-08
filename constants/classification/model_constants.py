from pathlib import Path
from enum import Enum

class ModelConstants:
    RELU_ACTIVATION_FUNCTION='relu'
    OUTPUT_ACTIVATION = 'softmax'
    PADDING_SAME = 'same'
    LOSS_CATEGORICAL_CROSSENTROPY = 'categorical_crossentropy'
    LOSS_METRIC = 'loss'
    METRIC_ACCURACY = 'accuracy'
    METRIC_PRECISION = 'precision'
    METRIC_RECALL = 'recall'
    METRIC_AUC = 'auc'
    EPOCHS = 100
    STATE_MODEL_WEIGHTS = 'model_weights'
    STATE_MODEL_CONFIG = 'model_config'
    CHECKPOINT_DIR_PATH = Path('Checkpoints') / 'ClassificationModel'
    CANCER_TYPE_RESULT_KEY = 'cancer_type'
    CONFIDENCE_KEY = 'confidence'
    MODEL_NAME = 'Cancer Classification Model'
    TRACKED_LOSS_METRIC = 'loss'
    TRACKED_VAL_LOSS_METRIC = 'val_loss'
    TRACKED_ACCURACY_METRIC = 'accuracy'
    TRACKED_VAL_ACCURACY_METRIC = 'val_accuracy'
    HORIZONTAL_FLIP_AUGMENTATION = "horizontal"
