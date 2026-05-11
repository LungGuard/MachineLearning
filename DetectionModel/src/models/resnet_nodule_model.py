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
    def __init__(self,
                 learning_rate: float = 1e-4,
                 metrics: Union[dict, MetricCollection] = None,
                 weight_decay: float = 1e-4,
                 freeze_backbone=True,
                 dropout_p=0.2):
        super(ResNetNoduleModel, self).__init__()

        self.save_hyperparameters(ignore=[HyperParameters.METRICS])

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_fn = nn.MSELoss() 
        self.dropout_p = dropout_p

        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False  
        
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
        
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=self.dropout_p), 
            nn.Linear(num_ftrs, len(TARGET_FEATURES))
        )
        
        self._setup_metrics(metrics)

    def _default_metrics(self):
        return MetricCollection({
            Metrics.RMSE: torchmetrics.MeanSquaredError(squared=False),
            Metrics.MAE: torchmetrics.MeanAbsoluteError(),
            Metrics.R2: torchmetrics.R2Score(len(TARGET_FEATURES)),
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
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.1, 
            patience=10,
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