from enum import StrEnum


class YoloVariant(StrEnum):
    YOLO_NANO = "yolov8n.pt"
    YOLO_SMALL = "yolov8s.pt"
    YOLO_MEDIUM = "yolov8m.pt"
    YOLO_LARGE = "yolov8l.pt"
    YOLO_EXTRA_LARGE = "yolov8x.pt"
