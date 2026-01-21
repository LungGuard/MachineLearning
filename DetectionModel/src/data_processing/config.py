"""
Data Preparation Configuration
================================
Configuration dataclass for the offline data preparation pipeline.

Author: LungGuard ML Team
License: Proprietary
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class DataPrepConfig:
    """Configuration for data preparation pipeline."""
    
    # Paths
    data_path: str = "/data/LIDC-IDRI"
    output_dir: str = "./lungguard_dataset"
    
    # Split ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Image parameters
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    window_center: float = -600.0
    window_width: float = 1500.0
    
    # YOLO parameters
    bbox_padding_factor: float = 1.5
    class_id: int = 0  # 0 = nodule/anomaly
    
    # Processing parameters
    min_nodule_diameter: float = 3.0  # mm, minimum nodule size to include
    max_nodule_diameter: float = 100.0  # mm, maximum nodule size
    slices_per_nodule: int = 3  # Number of slices to generate per nodule
    
    # Random seed for reproducibility
    random_seed: int = 42
