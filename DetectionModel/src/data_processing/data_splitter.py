"""
Data Splitting Utilities
=========================
Patient-level data splitting to prevent data leakage.

This module ensures that all data from a single patient stays in one split
(train, validation, or test) to maintain proper evaluation integrity.

Author: LungGuard ML Team
License: Proprietary
"""

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
    """
    Split patient IDs into train/val/test sets.
    
    CRITICAL: Split by patient ID to prevent data leakage.
    A patient's data must exist in ONLY one split.
    
    Parameters
    ----------
    patient_ids : List[str]
        List of unique patient identifiers
    train_ratio : float
        Proportion of data for training (e.g., 0.70)
    val_ratio : float
        Proportion of data for validation (e.g., 0.15)
    test_ratio : float
        Proportion of data for testing (e.g., 0.15)
    random_seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    Dict[str, List[str]]
        Dictionary with 'train', 'val', 'test' keys mapping to patient IDs
    """
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
    """
    Determine which split a patient belongs to.
    
    Parameters
    ----------
    patient_id : str
        Patient identifier
    splits : Dict[str, List[str]]
        Split dictionary from split_patients_by_id
    
    Returns
    -------
    str
        Split name ('train', 'val', or 'test')
    """
    split_mapping = {
        pid: split_name
        for split_name, pids in splits.items()
        for pid in pids
    }
    
    result = split_mapping.get(patient_id, 'train')  # Default to train if not found
    
    return result
