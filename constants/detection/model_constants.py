
class DetectionModelConstants:
    MODEL_NAME = "Detection Model"
    YOLOV8_BACKBONE_PRESET = "yolo_v8_l_backbone_coco" # 'n', 's', 'm', 'l', 'x' variants are also available
    BOUNDING_BOX_FORMAT= "xywh"
    CLASSIFICATION_LOSS='binary_crossentropy'
    BOX_LOSS='ciou'
    DEFAULT_INPUT_SHAPE = (640, 640, 3)
    NUM_CLASSES=1
    EPOCHES = 100
    DEFAULT_LEARNING_RATE=1e-3

class RegressionModelConstants:
    pass