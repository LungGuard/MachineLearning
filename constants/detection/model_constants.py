from enum import Enum
from pathlib import Path

class DetectionModelConstants:
    PROJECT_ROOT = Path(__file__).parents[2]
    DATASET_DIR=PROJECT_ROOT / "datasets"
    MODEL_NAME = "Detection Model"
    LEARNING_RATE = 0.001
    BATCH_SIZE = 16 
    DATASET_YAML = DATASET_DIR/ "data.yaml"  

    DEFAULT_INPUT_SIZE=(3, 640, 640)
    DEFAULT_CONFIDENCE_THRESHOLD=0.25

    class NativeTrainingArgs:
        DATA_ARG_NAME="data"
        EPOCHS_ARG_NAME = "epochs"
        IMAGES_ARG_NAME = "imgsz"
        BATCH_ARG_NAME = "batch"
        DEVICE_ARG_NAME = "device"
        PROJECT_ARG_NAME = "project"
        MODEL_NAME_ARG_NAME = "name"
        EXIST_OK_ARG_NAME = "exist_ok"
        VERBOSE_ARG_NAME = "verbose"
        LR0_ARG_NAME="lr0"
    class Results:
        BOUNDING_BOXES_KEY = "bounding_boxes"
        CONFIDENCE_SCORES_KEY = "confidence_scores"
        NODULES_COUNT="nodules_count"


class YoloVariant(Enum):
    YOLO_NANO = ("yolov8n.pt", 1, 5)
    YOLO_SMALL = ("yolov8s.pt", 2, 4)
    YOLO_MEDIUM = ("yolov8m.pt", 3, 3)
    YOLO_LARGE = ("yolov8l.pt", 4, 2)
    YOLO_EXTRA_LARGE= ("yolov8x.pt",5,1)

    def __init__(self, preset, accuracy, speed):
        self.preset = preset
        self.accuracy = accuracy
        self.speed = speed

    @property
    def is_medical_grade(self):
        return self.accuracy >= 4

    @classmethod
    def get_medical_grade_variants(cls):
        return [variant for variant in cls if variant.is_medical_grade]

