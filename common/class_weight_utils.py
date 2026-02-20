"""Utilities for calculating class weights to handle imbalanced datasets."""

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import logging

logger = logging.getLogger(__name__)


def calculate_class_weights(dataset, class_names=None, verbose=True):
    """
    Calculate balanced class weights for handling imbalanced datasets.
    Returns a Dictionary mapping class indices to their weights
    """
    # Extract all labels from the dataset
    labels = []
    for _, batch_labels in dataset:
        # Convert one-hot encoded labels to class indices
        labels.extend(np.argmax(batch_labels.numpy(), axis=1))

    labels = np.array(labels)
    classes = np.unique(labels)

    # Compute balanced weights: weight = n_samples / (n_classes * n_samples_per_class)
    weights_array = compute_class_weight('balanced', classes=classes, y=labels)
    class_weights = dict(enumerate(weights_array))

    if verbose:
        logger.debug("Class weights for handling imbalance:")
        for idx, weight in class_weights.items():
            class_label = class_names[idx] if class_names else f"Class {idx}"
            print(f"  {class_label}: {weight:.3f}")

    return class_weights
