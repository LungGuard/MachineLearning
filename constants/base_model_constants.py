import torch


class BaseModelConstants:
    METRICS_HYPERPARAMETER='metrics'
    CALLBACKS_HYPERPARAMETER='callbacks'
    LOSS_FN_HYPERPARAMETER="loss_fn"
    DEFAULT_METRICS_TASK = "multiclass"
    EVAL_TRAINER_ACCELERATOR_MODE="auto"
    DEFAULT_LEARNING_RATE =  1e-3
    DEFAULT_OPTIMIZER = torch.optim.Adam
