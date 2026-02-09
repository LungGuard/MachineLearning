from enum import Enum
from pathlib import Path

class DetectionModelConstants:
    PROJECT_ROOT = Path(__file__).parents[2]
    DATASET_DIR=PROJECT_ROOT / "datasets"
    MODEL_NAME = "Detection Model"
    LEARNING_RATE = 0.001
    BATCH_SIZE = 16 
    DATASET_YAML = DATASET_DIR/ "data.yaml"  



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

