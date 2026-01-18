from ClassificationModel.src.data_processing.merge_datasets import DatasetMerger
import os
from constants.classification.datasets_constants import DatasetConstants
import tensorflow as tf
from ClassificationModel.src.data_processing.image_augmentation import ImageAugmentationPipeline

def load_dataset(base_path=DatasetConstants.UNIFIED_DATASET_DIR
                ,image_size=DatasetConstants.IMAGE_SIZE
                ,batch_size=DatasetConstants.BATCH_SIZE):
    '''
    Returns a dict consists of a split name key and a dataset value.
    Should be used only for training due to the nature of the tensorflow helper methods
    '''
    if DatasetConstants.UNIFIED_DATASET_NAME not in os.listdir(DatasetConstants.DATASETS_DIR):
        merger = DatasetMerger(
            figshare_dir=DatasetConstants.FIGSHARE_DIR,
            huggingface_cache=DatasetConstants.HUGGINGFACE_CACHE,
            output_dir=DatasetConstants.UNIFIED_DATASET_DIR
        )
        merger.merge()
    dataset = {}

    train = tf.keras.utils.image_dataset_from_directory(
        f'{base_path}/{DatasetConstants.TRAIN_SPLIT_NAME}',
        image_size=image_size,
        batch_size=batch_size,
        label_mode=DatasetConstants.DATASET_LABEL_MODE,
        color_mode='grayscale',
        shuffle=True,
        seed=DatasetConstants.SEED
    )
    validation = tf.keras.utils.image_dataset_from_directory(
        f'{base_path}/{DatasetConstants.VAL_SPLIT_NAME}',
        image_size=image_size,
        batch_size=batch_size,
        label_mode=DatasetConstants.DATASET_LABEL_MODE,
        color_mode='grayscale',
        shuffle=False
    )
    
    test= tf.keras.utils.image_dataset_from_directory(
        f'{base_path}/{DatasetConstants.TEST_SPLIT_NAME}',
        image_size=image_size,
        batch_size=batch_size,
        label_mode=DatasetConstants.DATASET_LABEL_MODE,
        color_mode='grayscale',
        shuffle=False
    )
    class_names = train.class_names
    num_classes = len(class_names)

    dataset = {
        DatasetConstants.TRAIN_SPLIT_NAME: train,
        DatasetConstants.TEST_SPLIT_NAME: test,
        DatasetConstants.VAL_SPLIT_NAME: validation,
        DatasetConstants.CLASS_NAMES_KEY : class_names,
        DatasetConstants.NUM_CLASSES_KEY :num_classes
    }
    return dataset







    
    
    
    