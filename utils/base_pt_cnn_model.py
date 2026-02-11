import torch
import torch.nn as nn
import lightning as L
from pathlib import Path
from torchmetrics import MetricCollection, Accuracy
from utils.notification_service import NtfyNotificationService
from constants.base_model_constants import BaseModelConstants
from constants.common.metrics_constants import Metrics
from constants.common.model_stages import ModelStage
import logging

from .pt_layers.Conv2D_block import Conv2DBlock
from .pt_layers.DenseBlock import DenseBlock


logger = logging.getLogger(__name__)


class BaseCNNModel(L.LightningModule):

    """Base class for all CNN models"""

    def __init__(self, 
                 input_shape: tuple, 
                 model_name: str, 
                 num_classes: int,
                 learning_rate: float =BaseModelConstants.DEFAULT_LEARNING_RATE,
                 loss_fn: nn.Module = None,
                 metrics=None,
                 optimizer_cls=BaseModelConstants.DEFAULT_OPTIMIZER,
                 additional_optimizers=None,
                 callbacks=None):
        
        super().__init__()
        self.save_hyperparameters(ignore=[
            BaseModelConstants.METRICS_HYPERPARAMETER, 
            BaseModelConstants.CALLBACKS_HYPERPARAMETER
        ])

        self.model_name = model_name
        self.channels, self.height, self.width = input_shape
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.optimizer_cls = optimizer_cls
        self.additional_optimizers = additional_optimizers or {}
        self.custom_callbacks = callbacks or []
        self.features = nn.ModuleList()

        self._build_model()
        self._setup_metrics(metrics)

    # ======================== ABSTRACT ========================

    def _build_model(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    # ======================== METRICS ========================

    def _setup_metrics(self, metrics):
        collection = self._create_metric_collection(metrics)
        self.train_metrics = collection.clone(prefix=ModelStage.TRAIN.prefix)
        self.val_metrics = collection.clone(prefix=ModelStage.VAL.prefix)
        self.test_metrics = collection.clone(prefix=ModelStage.TEST.prefix)

    def _create_metric_collection(self, metrics):
        
        is_none = metrics is None
    
        default_metrics = MetricCollection({
            Metrics.METRIC_ACCURACY.value: Accuracy(
                task=BaseModelConstants.DEFAULT_METRICS_TASK, 
                num_classes=self.num_classes
            )
        })

        is_dict = isinstance(metrics, dict)
        result = default_metrics
        result = MetricCollection(metrics) if is_dict else result
        result = default_metrics if is_none else result
        return result

    # ======================== OPTIMIZER ========================

    def configure_optimizers(self):
        return self.optimizer_cls(
            self.parameters(), 
            lr=self.learning_rate, 
            **self.additional_optimizers
        )

    def configure_callbacks(self):
        return self.custom_callbacks

    # ======================== SHARED STEP LOGIC ========================

    def _compute_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    # ======================== TRAINING ========================

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self._common_step(batch,self.train_metrics)
        self.log(Metrics.DEFAULT_METRIC_LOSS.get_metric_variant(ModelStage.TRAIN), loss,on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # ======================== VALIDATION ========================

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._common_step(batch,self.val_metrics)
        self.log(Metrics.DEFAULT_METRIC_LOSS.get_metric_variant(ModelStage.VAL), loss, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    # ======================== TEST ========================

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self._common_step(batch,self.test_metrics)
        self.log(Metrics.DEFAULT_METRIC_LOSS.get_metric_variant(ModelStage.TEST), loss)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)
        return loss
    
    def _common_step(self,batch,batch_idx,model_stage_metrics):
        loss, preds, targets = self._compute_step(batch)
        model_stage_metrics.update(preds, targets)
        return loss,preds,targets
        


    # ======================== LOADING ========================


    def _get_conv_block(self, in_channels, out_channels, kernel_size=3):
        return Conv2DBlock(in_channels, out_channels, kernel_size, nn.ReLU())

    def _get_dense_block(self, in_features, out_features):
        return DenseBlock(in_features, out_features, nn.ReLU())
    