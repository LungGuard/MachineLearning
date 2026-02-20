from enum import StrEnum


class Accelerator(StrEnum):
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    AUTO = "auto"
    APPLE = "mps"
