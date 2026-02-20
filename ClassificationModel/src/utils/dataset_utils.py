from ClassificationModel.src.data_processing.merge_datasets import DatasetMerger
import os
from ClassificationModel.constants.constants.dataset import DatasetConstants
import tensorflow as tf
from ClassificationModel.src.data_processing.image_augmentation import ImageAugmentationPipeline
import torch
from dataclasses import dataclass
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import List
from tqdm import tqdm

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



@dataclass
class LungCancerDataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: List[str]
    num_classes: int
    # We save the calculated stats in case you need them later
    mean: torch.Tensor = None 
    std: torch.Tensor = None

class LungCancerDataManager:
    def __init__(self, 
                 base_path=DatasetConstants.UNIFIED_DATASET_DIR,
                 image_size=DatasetConstants.IMAGE_SIZE,
                 batch_size=DatasetConstants.BATCH_SIZE,
                 num_workers=2):
        self.base_path = base_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _ensure_dataset_ready(self):
        """Checks if data exists, merges if not."""
        if not os.path.exists(DatasetConstants.DATASETS_DIR) or \
           DatasetConstants.UNIFIED_DATASET_NAME not in os.listdir(DatasetConstants.DATASETS_DIR):
            print("Unified dataset not found. Merging datasets...")
            merger = DatasetMerger(
                figshare_dir=DatasetConstants.FIGSHARE_DIR,
                huggingface_cache=DatasetConstants.HUGGINGFACE_CACHE,
                output_dir=DatasetConstants.UNIFIED_DATASET_DIR
            )
            merger.merge()

    def _calculate_statistics(self, dataset: datasets.ImageFolder):
        """
        Internal helper: Iterates over the dataset to find Mean and Std.
        """
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
        print("Adapting: Calculating dataset statistics for Normalization...")
        cnt = 0
        fst_moment = torch.empty(3)
        snd_moment = torch.empty(3)

        for images, _ in tqdm(loader, desc="Scanning"):
            images = images.to(self.device)
            b, c, h, w = images.shape
            nb_pixels = b * h * w
            
            sum_ = torch.sum(images, dim=[0, 2, 3])
            sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
            
            if cnt == 0:
                fst_moment = sum_
                snd_moment = sum_of_square
            else:
                fst_moment = fst_moment + sum_
                snd_moment = snd_moment + sum_of_square
            cnt += nb_pixels

        mean = fst_moment / cnt
        std = torch.sqrt(snd_moment / cnt - mean ** 2)
        
        print(f"Calculated -> Mean: {mean.cpu().numpy()}, Std: {std.cpu().numpy()}")
        return mean.cpu(), std.cpu()

    def load(self, automatic_normalization=True) -> LungCancerDataBundle:
        """
        Loads the dataset. 
        If automatic_normalization=True, it acts like .adapt():
        it scans the training set, calculates stats, and applies normalization to ALL splits.
        """
        self._ensure_dataset_ready()

        # 1. Define Base Transforms (Without Normalization)
        base_transforms_list = [
            transforms.Resize(self.image_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
        
        # 2. Create a temporary training dataset to calculate stats
        train_path = os.path.join(self.base_path, DatasetConstants.TRAIN_SPLIT_NAME)
        temp_transform = transforms.Compose(base_transforms_list)
        temp_train_ds = datasets.ImageFolder(root=train_path, transform=temp_transform)

        mean, std = None, None
        
        # 3. Calculate Mean/Std if requested
        if automatic_normalization:
            mean, std = self._calculate_statistics(temp_train_ds)
            # Add Normalize to the transform list
            # This ensures output data is (pixel - mean) / std
            base_transforms_list.append(transforms.Normalize(mean=mean, std=std))

        # 4. Create the Final Transform Pipeline
        final_transform = transforms.Compose(base_transforms_list)

        # 5. Create Final Datasets (using the normalized transform)
        train_ds = datasets.ImageFolder(root=train_path, transform=final_transform)
        
        val_ds = datasets.ImageFolder(
            root=os.path.join(self.base_path, DatasetConstants.VAL_SPLIT_NAME),
            transform=final_transform
        )
        
        test_ds = datasets.ImageFolder(
            root=os.path.join(self.base_path, DatasetConstants.TEST_SPLIT_NAME),
            transform=final_transform
        )

        # 6. Create DataLoaders
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

        return LungCancerDataBundle(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_names=train_ds.classes,
            num_classes=len(train_ds.classes),
            mean=mean,
            std=std
        )