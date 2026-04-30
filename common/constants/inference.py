from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceConstants:
    CLASSIFICATION_INPUT_SIZE: tuple[int, int] = (224, 224)
    REGRESSION_INPUT_SIZE: tuple[int, int] = (64, 64)
    DEFAULT_MALIGNANCY_THRESHOLD: float = 3.0
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.5
    YOLO_PIXEL_NORMALIZER: float = 255.0
