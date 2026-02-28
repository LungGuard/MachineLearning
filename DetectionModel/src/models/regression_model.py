import lightning as L
import torch
import torch.nn as nn
import torchmetrics
from torchmetrics import MetricCollection
from typing import Union

from DetectionModel.constants.constants.regression_model import RegressionModelConstants
from DetectionModel.constants.dataclasses.nodule_features import NoduleFeatures
from DetectionModel.constants.enums.features import Features
from common.constants.metrics import Metrics
from common.constants.model_stages import ModelStage
from common.mixins import ModelMixin
from common.constants import HyperParameters

from common.layers.conv2d_block import Conv2DBlock
from common.layers.dense_block import DenseBlock

TAGRET_FEATURES = Features.getNoduleFeaturesVector()

class NoduleFeaturesModel(L.LightningModule,ModelMixin):
    def __init__(self,
                 input_shape = RegressionModelConstants.DEFAULT_INPUT_SHAPE,
                 learning_rate : float = RegressionModelConstants.DEFAULT_LEARNING_RATE,
                 metrics : Union[dict,MetricCollection] = None,
                 conv_layers_channels : Union[list[int],tuple[int]] = (32,64,128),
                 dense_layers_channels : Union[list[int],tuple[int]] =(128, 64)):
        super(NoduleFeaturesModel, self).__init__()

        self.save_hyperparameters(ignore=[HyperParameters.METRICS,
                                          HyperParameters.LAYERS])

        self.input_shape = input_shape
        self.channels, self.height, self.width = input_shape
        self.learning_rate = learning_rate

        self.feature_extractor = nn.Sequential()
        self.regressor = nn.Sequential()
        
        self.loss_fn = nn.MSELoss() 
        
        self._build_model(conv_layers=conv_layers_channels,
                          dense_layers=dense_layers_channels)
        self._setup_metrics(metrics)

    def _default_metrics(self):
        return MetricCollection({
            Metrics.RMSE: torchmetrics.MeanSquaredError(squared=False),
            Metrics.MAE: torchmetrics.MeanAbsoluteError(),
            Metrics.R2: torchmetrics.R2Score(len(TAGRET_FEATURES)),
        })

    def _build_model(self,conv_layers,dense_layers):
        self._add_chained_blocks(
            target=self.feature_extractor,
            channel_sizes=(self.channels,*conv_layers),
            name_prefix=RegressionModelConstants.CONV_BLOCK_NAME_PREFIX,
            block_class=Conv2DBlock
        )

        self._add_multiple_layers(target=self.feature_extractor,
                                  layers=[
                                    (RegressionModelConstants.BRIDGE_LAYER_NAME,nn.AdaptiveAvgPool2d((1, 1))),
                                    (RegressionModelConstants.FLATTEN_LAYER_NAME,nn.Flatten())
                                         ])

        self._add_chained_blocks(
        target=self.regressor,
        channel_sizes=dense_layers,
        name_prefix=RegressionModelConstants.DENSE_BLOCK_NAME_PREFIX,
        block_class=DenseBlock,
        )

        self.regressor.add_module(
                                  RegressionModelConstants.OUTPUT_LAYER_NAME,
                                  nn.Linear(64, len(TAGRET_FEATURES))
                                  ) 

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.regressor(features)

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
        self.log(Metrics.DEFAULT_LOSS.get_variant(ModelStage.TRAIN), loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, _, loss = self._common_step(batch, batch_idx, ModelStage.VAL)
        self.log(Metrics.DEFAULT_LOSS.get_variant(ModelStage.VAL), loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        _, _, loss = self._common_step(batch, batch_idx, ModelStage.TEST)
        self.log(Metrics.DEFAULT_LOSS.get_variant(ModelStage.TEST), loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.1, 
                patience=5, 
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": Metrics.DEFAULT_LOSS.get_variant(ModelStage.VAL), 
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