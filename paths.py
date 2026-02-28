"""
Cross-platform project paths — single source of truth.

All path constants derive from PROJECT_ROOT, which is resolved at import time
using `Path(__file__).resolve().parent`. Since this file lives at:
    MachineLearning/paths.py
PROJECT_ROOT always points to the MachineLearning workspace root,
regardless of OS or working directory.

Usage:
    from paths import ProjectPaths
    
    csv = ProjectPaths.DETECTION_METADATA_CSV
    dataset = ProjectPaths.CLASSIFICATION_UNIFIED_DIR / "train"
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


class ProjectPaths:
    """All project-wide directory and file paths."""

    ROOT = PROJECT_ROOT

    LOGS_DIR = ROOT / "logs"

    # ── Detection Model ──────────────────────────────────
    DETECTION_ROOT = ROOT / "DetectionModel"
    DETECTION_DATASETS_DIR = DETECTION_ROOT / "datasets_clean"
    DETECTION_DATASET_YAML = DETECTION_DATASETS_DIR / "dataset.yaml"
    DETECTION_METADATA_CSV = DETECTION_DATASETS_DIR / "metadata" / "regression_dataset.csv"
    DETECTION_CHECKPOINT_DIR = DETECTION_ROOT / "src" / "model_checkpoints"

    # ── Classification Model ─────────────────────────────
    CLASSIFICATION_ROOT = ROOT / "ClassificationModel"
    CLASSIFICATION_DATASETS_DIR = CLASSIFICATION_ROOT / "datasets"
    CLASSIFICATION_FIGSHARE_DIR = CLASSIFICATION_DATASETS_DIR / "figshare_dataset"
    CLASSIFICATION_HUGGINGFACE_CACHE = CLASSIFICATION_DATASETS_DIR / "hugging_face_dataset"
    CLASSIFICATION_UNIFIED_DIR = CLASSIFICATION_DATASETS_DIR / "unified_dataset_v2"
    CLASSIFICATION_CHECKPOINT_DIR = CLASSIFICATION_ROOT / "testing" / "Checkpoints"
    CLASSIFICATION_RESULTS_DIR = CLASSIFICATION_ROOT / "testing" / "results"
