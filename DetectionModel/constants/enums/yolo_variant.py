from enum import StrEnum


class YoloVariant(StrEnum):
    NANO = "yolov8n.pt"
    SMALL = "yolov8s.pt"
    MEDIUM = "yolov8m.pt"
    LARGE = "yolov8l.pt"
    EXTRA_LARGE = "yolov8x.pt"
