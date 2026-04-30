import re
from pathlib import Path
from paths import ProjectPaths

CHECKPOINT_DIR_PATH = ProjectPaths.DETECTION_CHECKPOINT_DIR

LOG_DIR_PATH = ProjectPaths.LOGS_DIR


def resolve_dataset_yaml(yaml_path: Path = ProjectPaths.DETECTION_DATASET_YAML) -> str:
    """
    Patch the ``path:`` field inside a YOLO dataset.yaml so that it contains the
    resolved absolute path of the yaml's parent directory.

    This is called at **training time** so the yaml always matches the current
    machine, regardless of OS.  The returned string can be passed straight to
    ``model.train(data=...)``.
    """
    yaml_path = Path(yaml_path).resolve()
    text = yaml_path.read_text()

    dataset_dir = str(yaml_path.parent).replace("\\", "/")
    new_text = re.sub(r"(?m)^path:.*$", f"path: {dataset_dir}", text)

    yaml_path.write_text(new_text)
    return str(yaml_path)


class DetectionModelConstants:
    PROJECT_ROOT = ProjectPaths.ROOT
    DATASET_DIR = ProjectPaths.DETECTION_DATASETS_DIR
    MODEL_NAME = "Detection Model"
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 32
    DATASET_YAML = ProjectPaths.DETECTION_DATASET_YAML
    LOG_DIR = ProjectPaths.LOGS_DIR
    MODEL_FILE_NAME = "detection_model_checkpoint"

    NMS_CONF_THRESHOLD = 0.25
    NMS_IOU_THRESHOLD = 0.45

    OPTIMIZER_WEIGHT_DECAY = 1e-3
    LR_SCHEDULER_FACTOR = 0.5
    LR_SCHEDULER_PATIENCE = 5

    DEFAULT_MODEL_YAML = "yolov8l.yaml"
    DEFAULT_PRETRAINED_WEIGHTS = "yolov8l.pt"
    NUM_CLASSES = 1
