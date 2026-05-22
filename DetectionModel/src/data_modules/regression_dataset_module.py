import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
import logging

from DetectionModel.constants.enums.features import Features
from DetectionModel.constants.enums.bbox import BBOX
from common.constants.model_stages import ModelStage
from DetectionModel.constants.dataclasses.transforms import TransformValues
from DetectionModel.constants.constants.dataset import DatasetConstants
from common.constants import HyperParameters

logger = logging.getLogger(__name__)


BBOX_COLUMNS = list(BBOX)

class AspectRatioPreservingResize:
    def __init__(self, target_size: int = DatasetConstants.DEFAULT_CROP_SIZE):
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

class NoduleRegressionDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        dataset_root: Path,
        target_features: list[str],
        crop_size: int = DatasetConstants.DEFAULT_CROP_SIZE,
        augment: bool = False,
        target_scaler: StandardScaler = None,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.dataset_root = Path(dataset_root)
        self.target_features = target_features
        self.crop_size = crop_size
        self.target_scaler = target_scaler

        self.crop_transform = AspectRatioPreservingResize(crop_size)
        self.transform_values = TransformValues()

        if augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=self.transform_values.horizontal_flip_probability),
                transforms.RandomVerticalFlip(p=self.transform_values.vertical_flip_probability),
                transforms.RandomRotation(degrees=self.transform_values.rotate_angle_range),
                transforms.ColorJitter(
                    brightness=self.transform_values.brightness_factor,
                    contrast=self.transform_values.contrast_factor,
                ),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            ])
            # RandomErasing is applied after to_tensor (operates on tensors)
            self.tensor_augment = transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
        else:
            self.augment_transform = None
            self.tensor_augment = None

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
        raw_targets = row[self.target_features].values.astype(np.float32).reshape(1, -1)
        if self.target_scaler is not None:
            raw_targets = self.target_scaler.transform(raw_targets).astype(np.float32)
        targets = torch.tensor(raw_targets.squeeze(0), dtype=torch.float32)
        return image, targets

    def _load_and_crop(self, row: pd.Series) -> torch.Tensor:
        image_rel = str(row[DatasetConstants.IMAGE_PATH]).replace("\\", "/")
        image_path = self.dataset_root / image_rel
        full_image = Image.open(image_path).convert("RGB")

        cropped = self._crop_nodule(full_image, row)
        resized = self.crop_transform(cropped)

        if self.augment_transform is not None:
            resized = self.augment_transform(resized)

        tensor = self.to_tensor(resized)
        if self.tensor_augment is not None:
            tensor = self.tensor_augment(tensor)
        return tensor

    def _crop_nodule(self, image: Image.Image, row: pd.Series,
                     margin_factor=DatasetConstants.MARGIN_FACTOR,
                     min_crop_size=DatasetConstants.MIN_CROP_SIZE) -> Image.Image:
        img_width, img_height = image.size

        x_center = row[BBOX.X] * img_width
        y_center = row[BBOX.Y] * img_height
        bbox_w = row[BBOX.W] * img_width
        bbox_h = row[BBOX.H] * img_height

        margin_x = bbox_w * margin_factor
        margin_y = bbox_h * margin_factor

        x_min = max(0, int(x_center - bbox_w / 2 - margin_x))
        y_min = max(0, int(y_center - bbox_h / 2 - margin_y))
        x_max = min(img_width, int(x_center + bbox_w / 2 + margin_x))
        y_max = min(img_height, int(y_center + bbox_h / 2 + margin_y))

        x_max = max(x_max, x_min + min_crop_size)
        y_max = max(y_max, y_min + min_crop_size)

        return image.crop((x_min, y_min, x_max, y_max))

class RegressionDataModule(L.LightningDataModule):
    def __init__(
        self,
        metadata_csv: Path,
        dataset_root: Path,
        target_features: list[str] = None,
        crop_size: int = DatasetConstants.DEFAULT_CROP_SIZE,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        augment_train = False,
        scale_targets: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[HyperParameters.TARGET_FEATURES])

        self.metadata_csv = Path(metadata_csv)
        self.dataset_root = Path(dataset_root)
        self.target_features = target_features or Features.getNoduleFeaturesVector()
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.augment_train = augment_train
        self.scale_targets = scale_targets
        self.target_scaler: StandardScaler = None

    @property
    def num_targets(self) -> int:
        return len(self.target_features)

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (3, self.crop_size, self.crop_size)

    def setup(self, stage = None) -> None:
        df = pd.read_csv(self.metadata_csv)
        self._validate_dataframe(df)

        split_map = {model_stage: df[df[DatasetConstants.SPLIT_GROUP] == model_stage]
                     for model_stage in ModelStage}

        self._log_split_stats(split_map)

        if self.scale_targets and self.target_scaler is None:
            train_targets = split_map[ModelStage.TRAIN][self.target_features].values.astype(np.float32)
            self.target_scaler = StandardScaler()
            self.target_scaler.fit(train_targets)
            # Cap scale_ for near-constant features so StandardScaler does not
            # artificially inflate their noise.  Any feature whose training std
            # is below the median std is clamped to the median std, which keeps
            # near-constant features (internal_structure std≈0.17) ihe same
            # numerical neighbourhood as the rest instead of blowing up to 1.
            median_scale = float(np.median(self.target_scaler.scale_))
            self.target_scaler.scale_ = np.maximum(self.target_scaler.scale_, median_scale)
            logger.info(
                f"StandardScaler fitted on {len(train_targets)} training samples "
                f"(low-variance features clamped to median scale={median_scale:.3f}). "
                f"Feature means: {dict(zip(self.target_features, self.target_scaler.mean_.round(3)))}"
            )

        splits_needed = self._resolve_splits(stage)
        for split_name in splits_needed:
            augment = self.augment_train and (split_name == ModelStage.TRAIN)
            dataset = NoduleRegressionDataset(
                dataframe=split_map[split_name],
                dataset_root=self.dataset_root,
                target_features=self.target_features,
                crop_size=self.crop_size,
                augment=augment,
                target_scaler=self.target_scaler,
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
        required = set(self.target_features + BBOX_COLUMNS + [DatasetConstants.SPLIT_GROUP,
                                                              DatasetConstants.IMAGE_PATH])
        missing = required - set(df.columns)
        assert not missing, f"Missing columns in CSV: {missing}"

    def _resolve_splits(self, stage) -> list:
        model_stages = list(ModelStage)
        stage_to_splits = {
            "fit": [ModelStage.TRAIN, ModelStage.VAL],
            "validate": [ModelStage.VAL],
            "test": [ModelStage.TEST],
            "predict": [ModelStage.TEST],
            None: model_stages,
        }
        return stage_to_splits.get(stage, model_stages)

    def _log_split_stats(self, split_map) -> None:
        for stage_enum, split_df in split_map.items():
            if split_df.empty:
                continue
            target_means = split_df[self.target_features].mean()
            logger.info(
                f"{stage_enum.prefix}: {len(split_df)} samples | "
                f"malignancy mean={target_means[Features.MALIGNANCY.value]:.2f}, "
                f"spiculation mean={target_means[Features.SPICULATION.value]:.2f}"
            )