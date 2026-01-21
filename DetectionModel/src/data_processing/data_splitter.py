"""Patient-level data splitting."""

import logging
from typing import List, Dict
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def split_patients_by_id(
    patient_ids: List[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int = 42
) -> Dict[str, List[str]]:
    """Split patient IDs into train/val/test sets."""
    # First split: train vs (val + test)
    train_ids, temp_ids = train_test_split(
        patient_ids,
        train_size=train_ratio,
        random_state=random_seed
    )
    
    # Second split: val vs test (from remaining)
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_ids, test_ids = train_test_split(
        temp_ids,
        train_size=val_ratio_adjusted,
        random_state=random_seed
    )
    
    splits = {
        'train': list(train_ids),
        'val': list(val_ids),
        'test': list(test_ids)
    }
    
    logger.info(
        f"Patient split - Train: {len(train_ids)}, "
        f"Val: {len(val_ids)}, Test: {len(test_ids)}"
    )
    
    return splits


def get_patient_split(patient_id: str, splits: Dict[str, List[str]]) -> str:
    """Get which split a patient belongs to."""
    split_mapping = {
        pid: split_name
        for split_name, pids in splits.items()
        for pid in pids
    }
    
    result = split_mapping.get(patient_id, 'train')  # Default to train if not found
    
    return result
