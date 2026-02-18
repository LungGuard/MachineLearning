from enum import Enum,StrEnum
from pathlib import Path
from dataclasses import dataclass, asdict
import torch


class DetectionModelConstants:
    PROJECT_ROOT = Path(__file__).parents[2]
    DATASET_DIR=PROJECT_ROOT / "datasets"
    MODEL_NAME = "Detection Model"
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 16 
    DATASET_YAML = DATASET_DIR/ "data.yaml"  
    
    class Results:
        BOUNDING_BOXES_KEY = "bounding_boxes"
        CONFIDENCE_SCORES_KEY = "confidence_scores"
        NODULES_COUNT="nodules_count"

class RegressionModelConstants:
    DEFAULT_INPUT_SHAPE=(3,64,64)
    DEFAULT_LEARNING_RATE=1e-4
    CONV_BLOCK_NAME_PREFIX='conv'
    DENSE_BLOCK_NAME_PREFIX='dense'
    BRIDGE_LAYER_NAME='bridge'
    FLATTEN_LAYER_NAME='flatten'
    OUTPUT_LAYER_NAME='output'

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




 


