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

TARGET_FEATURES = Features.getNoduleFeaturesVector()


class ResNetNoduleModel(L.LightningModule, ModelMixin):
    """
    ResNet18-based regression model for nodule feature prediction.

    Freezing strategy (applied in order):
        freeze_backbone=True  → freeze all backbone layers
        head_only=False       → then unfreeze layer3 + layer4 (partial fine-tuning)
        head_only=True        → keep ALL backbone frozen (only head trains)

    Head architecture:
        num_ftrs → head_hidden → (head_hidden//2) → len(TARGET_FEATURES)
        Each hidden layer: Linear + BN + ReLU + Dropout
    """

    def __init__(self,
                 learning_rate: float = 1e-4,
                 backbone_lr_factor: float = 0.1,
                 metrics: Union[dict, MetricCollection] = None,
                 weight_decay: float = 1e-4,
                 freeze_backbone: bool = True,
                 head_only: bool = False,
                 dropout_p: float = 0.3,
                 head_hidden: int = 256,
                 plat_patience: int = 10,
                 plat_factor: float = 0.5):
        super(ResNetNoduleModel, self).__init__()

        self.save_hyperparameters(ignore=[HyperParameters.METRICS])

        self.learning_rate = learning_rate
        self.backbone_lr_factor = backbone_lr_factor
        self.weight_decay = weight_decay
        self.dropout_p = dropout_p
        self.head_hidden = head_hidden
        self.plat_patience = plat_patience
        self.plat_factor = plat_factor
        self.loss_fn = nn.MSELoss()

        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if not head_only:
            for param in self.backbone.layer3.parameters():
                param.requires_grad = True
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True

        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = self._build_head(num_ftrs, head_hidden)

        self._setup_metrics(metrics)

    def _build_head(self, num_ftrs: int, hidden: int) -> nn.Sequential:
        if hidden > 0:
            return nn.Sequential(
                nn.Linear(num_ftrs, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(hidden, hidden // 2),
                nn.BatchNorm1d(hidden // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.dropout_p),
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
            if p.requires_grad and 'fc' not in name
        ]
        head_params = list(self.backbone.fc.parameters())

        param_groups = [
            {'params': head_params, 'lr': self.learning_rate},
        ]
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': self.learning_rate * self.backbone_lr_factor,
            })

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)

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

    def predict_features(self, x):
        was_training = self.training
        self.eval()
        with torch.inference_mode():
            raw_vectors = self(x).cpu()
        self.train(was_training)
        return [NoduleFeatures.from_tensor(p) for p in raw_vectors]
