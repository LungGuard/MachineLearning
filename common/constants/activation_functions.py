from enum import StrEnum


class Activation(StrEnum):
    RELU = "relu"
    LEAKY_RELU = 'leaky_relu'
    SOFTMAX = 'softmax'