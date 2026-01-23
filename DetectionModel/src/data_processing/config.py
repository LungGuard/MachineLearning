"""Data preparation configuration."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class DataPrepConfig:
    """Pipeline configuration parameters."""
    
    data_path: str = "/data/LIDC-IDRI"
    output_dir: str = "./lungguard_dataset"
    
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    window_center: float = -600.0
    window_width: float = 1500.0
    
    bbox_padding_factor: float = 1.5
    class_id: int = 0  # 0 = nodule/anomaly
    
    
    min_nodule_diameter: float = 3.0  #mm
    max_nodule_diameter: float = 100.0  #mm
    slices_per_nodule: int = 3 
    
    random_seed: int = 42
    log_freq:int = 5
