from enum import Enum,StrEnum
from pathlib import Path
from dataclasses import dataclass, asdict
import torch
import os
PROJECT_ROOT_DIR = Path(__file__).parents[2]

CHECKPOINT_DIR_PATH = PROJECT_ROOT_DIR / "DetectionModel" / "src" / "model_checkpoints"

LOG_DIR_PATH=PROJECT_ROOT_DIR/ "logs"

class DetectionModelConstants:
    PROJECT_ROOT = PROJECT_ROOT_DIR
    DATASET_DIR=PROJECT_ROOT / "datasets"
    MODEL_NAME = "Detection Model"
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 16 
    DATASET_YAML = DATASET_DIR/ "data.yaml"  
    LOG_DIR=LOG_DIR_PATH
class RegressionModelConstants:
    MODEL_NAME = "Nodule Features Prediction Model" 
    PROJECT_ROOT = PROJECT_ROOT_DIR
    DEFAULT_INPUT_SHAPE = (3,64,64)
    DEFAULT_LEARNING_RATE = 1e-4
    CONV_BLOCK_NAME_PREFIX = 'conv'
    DENSE_BLOCK_NAME_PREFIX = 'dense'
    BRIDGE_LAYER_NAME = 'bridge'
    FLATTEN_LAYER_NAME = 'flatten'
    OUTPUT_LAYER_NAME = 'output'
    CHECKPOINT_DIR=CHECKPOINT_DIR_PATH
    EXCEPTION_CHECKPOINT_FILE_NAME = "exception_reg_model_checkpoint"
    BEST_MODEL_CHECKPOINT_NAME = "best_reg_model"
    LOG_DIR=LOG_DIR_PATH

class YoloVariant(StrEnum):
    YOLO_NANO = "yolov8n.pt"
    YOLO_SMALL = "yolov8s.pt"
    YOLO_MEDIUM = "yolov8m.pt"
    YOLO_LARGE = "yolov8l.pt"
    YOLO_EXTRA_LARGE= "yolov8x.pt"

@dataclass(frozen=True)
class NoduleFeatures:
    malignancy: float
    spiculation: float
    lobulation: float
    subtlety: float
    sphericity: float
    margin: float
    texture: float
    calcification: float
    
    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor):

        values = tensor.detach().cpu().numpy().flatten()
        return cls(
            malignancy=float(values[0]),
            spiculation=float(values[1]),
            lobulation=float(values[2]),
            subtlety=float(values[3]),
            sphericity=float(values[4]),
            margin=float(values[5]),
            texture=float(values[6]),
            calcification=float(values[7])
        )

class Accelerator(StrEnum):
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    AUTO = "auto"
    APPLE = "mps"
