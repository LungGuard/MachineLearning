import torch


class BaseModelConstants:
    METRICS_HYPERPARAMETER='metrics'
    CALLBACKS_HYPERPARAMETER='callbacks'
    DEFAULT_METRICS_TASK = "multiclass"
    EVAL_TRAINER_ACCELERATOR_MODE="auto"
    TRAIN_PREFIX="train_"
    DEFAULT_LEARNING_RATE =  1e-3
    DEFAULT_OPTIMIZER = torch.optim.Adam
