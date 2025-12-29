from data_processing.merge_datasets import DatasetMerger
import os
from common.constants.datasets_constants import DatasetConstants
import tensorflow as tf

def get_dataset(base_path=DatasetConstants.UNIFIED_DATASET_DIR
                ,image_size=DatasetConstants.IMAGE_SIZE
                ,batch_size=DatasetConstants.BATCH_SIZE):
    "returns a dict consists of a split name key and a dataset value"
    if DatasetConstants.UNIFIED_DATASET_NAME not in os.listdir(DatasetConstants.DATASETS_DIR):
        merger = DatasetMerger(
            figshare_dir=DatasetConstants.FIGSHARE_DIR,
            huggingface_cache=DatasetConstants.HUGGINGFACE_CACHE,
            output_dir=DatasetConstants.UNIFIED_DATASET_DIR
        )
        merger.merge()
    dataset = {}

    
    
    
    