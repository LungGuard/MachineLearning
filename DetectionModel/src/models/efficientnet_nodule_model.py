import lightning as L
import torch
import torch.nn as nn
import torchvision.models as models
import torchmetrics
from torchmetrics import MetricCollection
from typing import Union

from DetectionModel.constants.constants.regression_model import RegressionModelConstants
from common.dto import NoduleFeatures
from DetectionModel.constants.enums.features import Features
from common.constants.metrics import Metrics
from common.constants.model_stages import ModelStage
from common.mixins import ModelMixin
from common.constants import HyperParameters, Loss
import torchvision.transforms.functional as TF

TARGET_FEATURES = Features.getNoduleFeaturesVector()


class EfficientNetNoduleModel(L.LightningModule, ModelMixin):
    """
    EfficientNet-B0 regression model for nodule feature prediction.

    EfficientNet-B0 has 9 feature blocks (indices 0-8).
    Freezing: freeze all, then unfreeze blocks [unfreeze_from:] + classifier head.
    Feature vector: 1280-dim after avgpool.

    Head: 1280 → head_hidden → (head_hidden//2) → len(TARGET_FEATURES)
    """

    def __init__(self,
                 learning_rate: float = 5e-4,
                 backbone_lr_factor: float = 0.1,
                 metrics: Union[dict, MetricCollection] = None,
                 weight_decay: float = 5e-3,
                 unfreeze_from: int = 7,
                 dropout_p: float = 0.3,
                 head_hidden: int = 256,
                 plat_patience: int = 10,
                 plat_factor: float = 0.5,
                 cosine_T0: int = 0,
                 cosine_T_mult: int = 2,
                 loss_fn: nn.Module = None):
        super(EfficientNetNoduleModel, self).__init__()

        self.save_hyperparameters(ignore=[HyperParameters.METRICS, 'loss_fn'])

        self.learning_rate = learning_rate
        self.backbone_lr_factor = backbone_lr_factor
        self.weight_decay = weight_decay
        self.dropout_p = dropout_p
        self.head_hidden = head_hidden
        self.plat_patience = plat_patience
        self.plat_factor = plat_factor
        self.cosine_T0 = cosine_T0
        self.cosine_T_mult = cosine_T_mult
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()

        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        for param in self.backbone.parameters():
            param.requires_grad = False

        for block_idx in range(unfreeze_from, len(self.backbone.features)):
            for param in self.backbone.features[block_idx].parameters():
                param.requires_grad = True

        num_ftrs = self.backbone.classifier[1].in_features
        self.backbone.classifier = self._build_head(num_ftrs, head_hidden)

        self._setup_metrics(metrics)

    def _build_head(self, num_ftrs: int, hidden: int) -> nn.Sequential:
        if hidden > 0:
            return nn.Sequential(
                nn.Dropout(p=self.dropout_p),
                nn.Linear(num_ftrs, hidden),
                nn.BatchNorm1d(hidden),
                nn.SiLU(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(hidden, hidden // 2),
                nn.BatchNorm1d(hidden // 2),
                nn.SiLU(),
                nn.Linear(hidden // 2, len(TARGET_FEATURES)),
            )
        return nn.Sequential(
            nn.Dropout(p=self.dropout_p),
            nn.Linear(num_ftrs, len(TARGET_FEATURES)),
        )

    def _default_metrics(self):
        return MetricCollection({
            Metrics.RMSE: torchmetrics.MeanSquaredError(squared=False),
            Metrics.MAE: torchmetrics.MeanAbsoluteError(),
            Metrics.R2: torchmetrics.R2Score(multioutput='variance_weighted'),
        })

    def forward(self, x):
        return self.backbone(x)

    def _common_step(self, batch, batch_idx, stage: ModelStage):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        metric_collection = self.model_stage_metrics[stage.prefix]
        metric_collection.update(y_pred, y)
        self.log_dict(metric_collection, on_step=False, on_epoch=True, prog_bar=True)
        return y_pred, y, loss

    def training_step(self, batch, batch_idx):
        _, _, loss = self._common_step(batch, batch_idx, ModelStage.TRAIN)
        self.log(Loss.DEFAULT.get_variant(ModelStage.TRAIN), loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, _, loss = self._common_step(batch, batch_idx, ModelStage.VAL)
        self.log(Loss.DEFAULT.get_variant(ModelStage.VAL), loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        _, _, loss = self._common_step(batch, batch_idx, ModelStage.TEST)
        self.log(Loss.DEFAULT.get_variant(ModelStage.TEST), loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        backbone_params = [
            p for name, p in self.backbone.named_parameters()
            if p.requires_grad and 'classifier' not in name
        ]
        head_params = list(self.backbone.classifier.parameters())

        param_groups = [
            {'params': head_params, 'lr': self.learning_rate},
        ]
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': self.learning_rate * self.backbone_lr_factor,
            })

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)

        if self.cosine_T0 > 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.cosine_T0,
                T_mult=self.cosine_T_mult,
                eta_min=1e-7,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.plat_factor,
            patience=self.plat_patience,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": Loss.DEFAULT.get_variant(ModelStage.VAL),
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def predict_features(self, x, tta: bool = True):
        """Run inference with optional test-time augmentation (TTA).

        TTA averages predictions over 5 mild augmented views:
        original, horizontal flip, vertical flip, +10° rotation, −10° rotation.
        This reduces prediction variance and improves R² by ~0.01–0.02 with no
        additional training cost.

        Args:
            x:   Input tensor (B, C, H, W) — already normalised.
            tta: If True, average over augmented views (recommended for inference).
        """

        was_training = self.training
        self.eval()

        with torch.inference_mode():
            if not tta:
                raw_vectors = self(x).cpu()
            else:
                views = [
                    x,
                    TF.hflip(x),
                    TF.vflip(x),
                    TF.rotate(x, angle=10),
                    TF.rotate(x, angle=-10),
                ]
                raw_vectors = sum(self(v) for v in views).div_(len(views)).cpu()

        self.train(was_training)
        return [NoduleFeatures.from_tensor(p) for p in raw_vectors]
