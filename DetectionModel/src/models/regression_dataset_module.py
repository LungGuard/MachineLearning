

import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
import logging
from constants.detection.features_enum import Features
from constants.detection.bbox_enum import BBOX
logger = logging.getLogger(__name__)



TARGET_FEATURES = [
    f.value for f in Features
]

BBOX_COLUMNS = [ bbox.value for bbox in BBOX ]
SPLIT_COLUMN = "split_group"
IMAGE_PATH_COLUMN = "image_path"

DEFAULT_CROP_SIZE = 64


# ─── Crop Transform ──────────────────────────────────────────────────────────

class AspectRatioPreservingResize:
    """
    Resize image preserving aspect ratio, then zero-pad to a square.

    Medical imaging rationale: nodule shape carries diagnostic information.
    Naive resize distorts aspect ratio — a spiculated 10×40px nodule becoming
    square loses the very feature we're trying to predict.

    Process:
        1. Scale longest edge to target_size
        2. Zero-pad shorter edge to make it square
    """

    def __init__(self, target_size: int = DEFAULT_CROP_SIZE):
        self.target_size = target_size

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = self.target_size / max(width, height)
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))

        resized = image.resize((new_width, new_height), Image.BILINEAR)

        padded = Image.new(resized.mode, (self.target_size, self.target_size), 0)
        paste_x = (self.target_size - new_width) // 2
        paste_y = (self.target_size - new_height) // 2
        padded.paste(resized, (paste_x, paste_y))

        return padded


# ─── Dataset ─────────────────────────────────────────────────────────────────

class NoduleRegressionDataset(Dataset):
    """
    Dataset that crops nodule regions from CT slices and pairs them with
    radiologist-consensus semantic features.

    Each sample:
        X: Cropped nodule image [3, crop_size, crop_size] (RGB/2.5D)
        Y: Semantic feature vector [9] (malignancy, spiculation, etc.)

    Bounding boxes in the CSV are normalized YOLO format (x_center, y_center, w, h)
    relative to the full image dimensions.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        dataset_root: Path,
        target_features: list[str],
        crop_size: int = DEFAULT_CROP_SIZE,
        augment: bool = False,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.dataset_root = Path(dataset_root)
        self.target_features = target_features
        self.crop_size = crop_size

        self.crop_transform = AspectRatioPreservingResize(crop_size)

        self.augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]) if augment else None

        self.to_tensor = transforms.ToTensor()

        logger.info(
            f"NoduleRegressionDataset: {len(self)} samples, "
            f"crop_size={crop_size}, augment={augment}, "
            f"targets={target_features}"
        )

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[idx]

        image = self._load_and_crop(row)
        targets = torch.tensor(
            row[self.target_features].values.astype(np.float32),
            dtype=torch.float32,
        )

        return image, targets

    def _load_and_crop(self, row: pd.Series) -> torch.Tensor:
        """Load full CT slice, crop nodule region using bbox, apply transforms."""
        image_path = self.dataset_root / row[IMAGE_PATH_COLUMN]
        full_image = Image.open(image_path).convert("RGB")

        cropped = self._crop_nodule(full_image, row)
        resized = self.crop_transform(cropped)

        augmented = (
            self.augment_transform(resized)
            if self.augment_transform is not None
            else resized
        )

        return self.to_tensor(augmented)

    def _crop_nodule(self, image: Image.Image, row: pd.Series) -> Image.Image:
        """
        Extract nodule region using normalized YOLO bounding box coordinates.

        YOLO format: (x_center, y_center, width, height) all in [0, 1]
        Converts to pixel coordinates and crops with a small margin for context.
        """
        img_width, img_height = image.size

        x_center = row["bbox_x"] * img_width
        y_center = row["bbox_y"] * img_height
        bbox_w = row["bbox_w"] * img_width
        bbox_h = row["bbox_h"] * img_height

        # 10% margin around nodule for surrounding tissue context
        margin_x = bbox_w * 0.1
        margin_y = bbox_h * 0.1

        x_min = max(0, int(x_center - bbox_w / 2 - margin_x))
        y_min = max(0, int(y_center - bbox_h / 2 - margin_y))
        x_max = min(img_width, int(x_center + bbox_w / 2 + margin_x))
        y_max = min(img_height, int(y_center + bbox_h / 2 + margin_y))

        # Ensure minimum crop of 4x4 pixels for very small nodules
        x_max = max(x_max, x_min + 4)
        y_max = max(y_max, y_min + 4)

        return image.crop((x_min, y_min, x_max, y_max))


# ─── DataModule ──────────────────────────────────────────────────────────────

class RegressionDataModule(L.LightningDataModule):
    """
    Lightning DataModule for the nodule regression CNN (Stage 1 risk scoring).

    Reads the metadata CSV, splits by 'split_group' column (patient-level split
    from data preparation), crops nodule regions from CT slices, and provides
    DataLoaders for training/validation/testing.

    Usage:
        dm = RegressionDataModule(
            metadata_csv=Path("DetectionModel/datasets/metadata/regression_dataset.csv"),
            dataset_root=Path("DetectionModel"),
            crop_size=64,
            batch_size=32,
        )
        dm.setup()

        # Access properties for model initialization
        print(dm.input_shape)   # (3, 64, 64)
        print(dm.num_targets)   # 9

        trainer = L.Trainer(...)
        trainer.fit(model, datamodule=dm)
    """

    def __init__(
        self,
        metadata_csv: Path,
        dataset_root: Path,
        target_features: list[str] = None,
        crop_size: int = DEFAULT_CROP_SIZE,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["target_features"])

        self.metadata_csv = Path(metadata_csv)
        self.dataset_root = Path(dataset_root)
        self.target_features = target_features or TARGET_FEATURES
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    @property
    def num_targets(self) -> int:
        """Number of output features for model initialization."""
        return len(self.target_features)

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Input tensor shape (C, H, W) for model initialization."""
        return (3, self.crop_size, self.crop_size)

    def setup(self, stage = None) -> None:
        """Load CSV, split by group, create datasets with cropping transforms."""
        df = pd.read_csv(self.metadata_csv)
        self._validate_dataframe(df)

        split_map = {
            "train": df[df[SPLIT_COLUMN] == "train"],
            "val": df[df[SPLIT_COLUMN] == "val"],
            "test": df[df[SPLIT_COLUMN] == "test"],
        }

        self._log_split_stats(split_map)

        splits_needed = self._resolve_splits(stage)
        for split_name in splits_needed:
            augment = split_name == "train"
            dataset = NoduleRegressionDataset(
                dataframe=split_map[split_name],
                dataset_root=self.dataset_root,
                target_features=self.target_features,
                crop_size=self.crop_size,
                augment=augment,
            )
            setattr(self, f"{split_name}_dataset", dataset)

    def train_dataloader(self) -> DataLoader:
        return self._build_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._build_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._build_dataloader(self.test_dataset, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    # ─── Private Helpers ─────────────────────────────────────────────────────

    def _build_dataloader(self, dataset: NoduleRegressionDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Verify all required columns exist."""
        required = set(self.target_features + BBOX_COLUMNS + [SPLIT_COLUMN, IMAGE_PATH_COLUMN])
        missing = required - set(df.columns)
        assert not missing, f"Missing columns in CSV: {missing}"

    def _resolve_splits(self, stage) -> list[str]:
        stage_to_splits = {
            "fit": ["train", "val"],
            "validate": ["val"],
            "test": ["test"],
            "predict": ["test"],
            None: ["train", "val", "test"],
        }
        return stage_to_splits.get(stage, ["train", "val", "test"])

    def _log_split_stats(self, split_map: dict[str, pd.DataFrame]) -> None:
        for name, split_df in split_map.items():
            target_means = split_df[self.target_features].mean()
            logger.info(
                f"{name}: {len(split_df)} samples | "
                f"malignancy mean={target_means['malignancy']:.2f}, "
                f"spiculation mean={target_means['spiculation']:.2f}"
            )