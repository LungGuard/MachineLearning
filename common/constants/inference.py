from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceConstants:
    CLASSIFICATION_INPUT_SIZE: tuple[int, int] = (224, 224)
    REGRESSION_INPUT_SIZE: tuple[int, int] = (224, 224)
    DEFAULT_MALIGNANCY_THRESHOLD: float = 3.0
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.5
    YOLO_PIXEL_NORMALIZER: float = 255.0
    # Number of slices per YOLO forward pass. Keeping this small limits peak RAM
    # usage: 8 slices × 3 × 512 × 512 × float32 ≈ 24 MB input, ~200 MB total.
    YOLO_INFERENCE_BATCH_SIZE: int = 8
