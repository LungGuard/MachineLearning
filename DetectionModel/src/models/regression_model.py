import lightning as L
import torch
import torch.nn as nn
import torchmetrics
from torchmetrics import MetricCollection

from DetectionModel.constants.constants.regression_model import RegressionModelConstants
from DetectionModel.constants.dataclasses.nodule_features import NoduleFeatures
from DetectionModel.constants.enums.features import Features
from common.constants.metrics import Metrics
from common.constants.model_stages import ModelStage
from common.mixins import ModelMixin

from common.layers.conv2d_block import Conv2DBlock
from common.layers.dense_block import DenseBlock


TARGET_FEATURES = [
    f for f in Features if f != Features.ANNOTATION_COUNT
] #Excluding the annotation count feature

class NoduleFeaturesModel(L.LightningModule,ModelMixin):
    def __init__(self,
                 input_shape = RegressionModelConstants.DEFAULT_INPUT_SHAPE,
                 learning_rate : float = RegressionModelConstants.DEFAULT_LEARNING_RATE,
                 metrics=None):
        super(NoduleFeaturesModel, self).__init__()
        
        self.save_hyperparameters() 

        self.input_shape = input_shape
        self.channels, self.height, self.width = input_shape
        self.learning_rate = learning_rate

        self.feature_extractor = nn.Sequential()
        self.regressor = nn.Sequential()
        
        self.loss_fn = nn.MSELoss() 
        
        self._build_model()
        self._setup_metrics(metrics)

    def _init_metrics(self, metrics):
        if metrics is not None:
            return MetricCollection(metrics) if isinstance(metrics, dict) else metrics
        else:
            return MetricCollection({
                Metrics.RMSE: torchmetrics.MeanSquaredError(squared=False),
                Metrics.MAE: torchmetrics.MeanAbsoluteError(),
                Metrics.R2: torchmetrics.R2Score(len(TARGET_FEATURES))
            })
    
    def _setup_metrics(self, metrics):
        base_metrics = self._init_metrics(metrics)

        self.model_stage_metrics = nn.ModuleDict({
            ModelStage.TRAIN.prefix: base_metrics.clone(prefix=ModelStage.TRAIN.prefix),
            ModelStage.VAL.prefix: base_metrics.clone(prefix=ModelStage.VAL.prefix),
            ModelStage.TEST.prefix: base_metrics.clone(prefix=ModelStage.TEST.prefix),
        })

    def _build_model(self):
        self._add_chained_blocks(
            target=self.feature_extractor,
            channel_sizes=[self.channels,32,64,128],
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
        channel_sizes=[128, 64],
        name_prefix=RegressionModelConstants.DENSE_BLOCK_NAME_PREFIX,
        block_class=DenseBlock,
        )

        self.regressor.add_module(RegressionModelConstants.OUTPUT_LAYER_NAME,
                                  nn.Linear(64, len(TARGET_FEATURES))) 

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
        self.eval() 

        with torch.no_grad():
            raw_vectors = self(x)
            raw_vectors=raw_vectors.cpu()
            
        return [NoduleFeatures.from_tensor(prediction) for prediction in raw_vectors]