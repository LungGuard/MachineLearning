"""Dataset output writer."""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

# Column name constant
SPLIT_COLUMN = 'split_group'


def _safe_count(df: Optional[pd.DataFrame], column: str, value: str) -> int:
    """Safely count rows matching a value."""
    has_data = df is not None and len(df) > 0 and column in df.columns
    return len(df[df[column] == value]) if has_data else 0


def save_metadata_csv(metadata_rows: List[Dict], output_path: Path) -> Optional[pd.DataFrame]:
    """Save metadata to CSV."""
    
    row_count = len(metadata_rows)
    logger.debug(f"Saving {row_count} metadata rows")
    
    metadata_df = pd.DataFrame(metadata_rows)
    
    logger.debug(f"Columns: {list(metadata_df.columns)}") if row_count > 0 else None
    
    logger.warning("No samples generated - CSV will be empty") if row_count == 0 else None
    
    metadata_df.to_csv(output_path, index=False)
    logger.info(f"Saved: {output_path} ({row_count} rows)")
    
    return metadata_df


def save_config_json(config, output_path: Path, metadata_df: Optional[pd.DataFrame]) -> Dict:
    """Save pipeline configuration."""
    
    config_dict = {
        'data_path': config.data_path,
        'output_dir': config.output_dir,
        'train_ratio': config.train_ratio,
        'val_ratio': config.val_ratio,
        'test_ratio': config.test_ratio,
        'target_spacing': config.target_spacing,
        'window_center': config.window_center,
        'window_width': config.window_width,
        'bbox_padding_factor': config.bbox_padding_factor,
        'min_nodule_diameter': config.min_nodule_diameter,
        'max_nodule_diameter': config.max_nodule_diameter,
        'slices_per_nodule': config.slices_per_nodule,
        'random_seed': config.random_seed,
        'generation_timestamp': datetime.now().isoformat(),
        'total_samples': len(metadata_df) if metadata_df is not None else 0,
        'train_samples': _safe_count(metadata_df, SPLIT_COLUMN, 'train'),
        'val_samples': _safe_count(metadata_df, SPLIT_COLUMN, 'val'),
        'test_samples': _safe_count(metadata_df, SPLIT_COLUMN, 'test')
    }
    
    with open(output_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info(f"Saved: {output_path}")
    return config_dict


def save_yolo_yaml(output_dir: str, metadata_df: Optional[pd.DataFrame]) -> Path:
    """Generate YOLO dataset.yaml."""
    
    total = len(metadata_df) if metadata_df is not None else 0
    train = _safe_count(metadata_df, SPLIT_COLUMN, 'train')
    val = _safe_count(metadata_df, SPLIT_COLUMN, 'val')
    test = _safe_count(metadata_df, SPLIT_COLUMN, 'test')
    
    yaml_content = f"""# LungGuard YOLO Dataset
path: {output_dir}
train: train/images
val: val/images
test: test/images

nc: 1
names:
  0: nodule

# Stats: {total} total ({train} train, {val} val, {test} test)
"""
    
    yaml_path = Path(output_dir) / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    logger.info(f"Saved: {yaml_path}")
    return yaml_path


def log_summary_statistics(
    metadata_df: Optional[pd.DataFrame],
    config_dict: Dict,
    csv_path: Path,
    config_path: Path,
    yaml_path: Path
):
    """Log final summary."""
    
    total = config_dict.get('total_samples', 0)
    
    logger.info("=" * 50)
    logger.info("COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Total samples: {total}")
    logger.info(f"  Train: {config_dict.get('train_samples', 0)}")
    logger.info(f"  Val:   {config_dict.get('val_samples', 0)}")
    logger.info(f"  Test:  {config_dict.get('test_samples', 0)}")
    logger.info(f"Files: {csv_path.name}, {config_path.name}, {yaml_path.name}")
    
    logger.warning(
        "\nNo samples generated! Check:\n"
        "  - DICOM path is correct\n"
        "  - Nodule diameter filters\n"
        "  - Run with --debug for details"
    ) if total == 0 else None