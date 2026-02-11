from pathlib import Path
from typing import Optional, List
import torch
import torch.nn as nn
import pandas as pd
import logging
from ultralytics import YOLO
import numpy as np
from torchvision.transforms.functional import to_tensor
from constants.detection.model_constants import DetectionModelConstants as ModelConstants, YoloVariant
from constants.common.metrics_constants import Metrics
from constants.common.model_stages import ModelStage
from utils.base_pt_cnn_model import BaseCNNModel


logger = logging.getLogger(__name__)


class NodulesDetectionModel(BaseCNNModel):
    """
    YOLO-based nodule detection for LungGuard Stage 1.
    
    Two training modes:
        1. YOLO-native: model.train_yolo_native() — uses Ultralytics trainer
        2. Lightning:   trainer.fit(model, loader) — uses Lightning loop
    """

    NODULE_CLASS_COUNT = 1

    def __init__(self, 
                 input_shape: tuple =ModelConstants.DEFAULT_INPUT_SIZE,
                 learning_rate: float = None,
                 additional_layers = None,
                 freeze_backbone: bool = True,
                 callbacks=None,
                 metrics=None,
                 **kwargs):

        self.additional_layers = additional_layers or []
        self.should_freeze_backbone = freeze_backbone
        self.yolo_model = None
        self.data_yaml_path = None

        super().__init__(
            input_shape=input_shape,
            model_name=ModelConstants.MODEL_NAME,
            num_classes=self.NODULE_CLASS_COUNT,
            learning_rate=learning_rate or ModelConstants.LEARNING_RATE,
            metrics=None,
            callbacks=callbacks,
            **kwargs
        )

    # ======================== MODEL BUILDING ========================

    def _build_model(self):
        self.yolo_model = YOLO(YoloVariant.YOLO_LARGE.preset)
        
        self._apply_backbone_freeze()
        self._apply_additional_layers()

    def _apply_backbone_freeze(self):
        if self.should_freeze_backbone:
            self._set_backbone_grad(requires_grad=False)

    def _apply_additional_layers(self):
        self.features.extend(self.additional_layers)
        layer_count = len(self.additional_layers)
        extra_msg = f" + {layer_count} custom layers" if layer_count > 0 else ""
        logger.info(f"Built: YOLO-{YoloVariant.YOLO_LARGE.preset}{extra_msg}")

    # ======================== FORWARD ========================

    def forward(self, x):
        out = self.yolo_model.model(x)

        for layer in self.features:
            out = layer(out)

        return out

    # ======================== LIGHTNING TRAINING ========================

    def training_step(self, batch, batch_idx):
        loss = self._compute_detection_loss(batch)
        self.log(Metrics.DEFAULT_METRIC_LOSS.get_model_stage_metric(ModelStage.TRAIN), loss, prog_bar=True)
        self.log_dict(self.train_metrics,prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_detection_loss(batch)
        self.log(Metrics.DEFAULT_METRIC_LOSS.get_model_stage_metric(ModelStage.VAL), loss, prog_bar=True, sync_dist=True)
        self.log_dict(self.train_metrics,prog_bar=True)
        return loss


    def _compute_detection_loss(self, batch):
        images, targets = batch
        raw_output = self.forward(images)
        return self.yolo_model.model.loss(raw_output, targets)


    def _parse_yolo_results(self, yolo_results) -> dict:
        csv_path = Path(yolo_results.save_dir) / "results.csv"
        empty_history = {Metrics.DEFAULT_METRIC_LOSS.get_metric_variant(ModelStage.TRAIN): [],
                         Metrics.DEFAULT_METRIC_LOSS.get_metric_variant(ModelStage.VAL): []}
        
        has_csv = csv_path.exists()
        df = pd.read_csv(csv_path) if has_csv else None
        
        result = empty_history
        
        parsed = self._extract_loss_columns(df) if has_csv else empty_history
        result = parsed if has_csv else result
        return result

    def _extract_loss_columns(self, df):
        df.columns = [col.strip() for col in df.columns]
        train_col = f"train/{Metrics.DEFAULT_METRIC_LOSS.get_metric_variant("box")}"
        val_col = f"val/{Metrics.DEFAULT_METRIC_LOSS.get_metric_variant("box")}"
        
        return {
            Metrics.DEFAULT_METRIC_LOSS.get_metric_variant(ModelStage.TRAIN): df[train_col].tolist() if train_col in df.columns else [],
            Metrics.DEFAULT_METRIC_LOSS.get_metric_variant(ModelStage.VAL): df[val_col].tolist() if val_col in df.columns else [],
        }

    def _replay_callbacks(self, callbacks, history):
        train_losses = history.get(Metrics.DEFAULT_METRIC_LOSS.get_metric_variant(ModelStage.TRAIN), [])
        val_losses = history.get(Metrics.DEFAULT_METRIC_LOSS.get_metric_variant(ModelStage.VAL), [])
        epoch_count = len(train_losses)

        for epoch_idx in range(epoch_count):
            train_metrics = {Metrics.DEFAULT_METRIC_LOSS.value : train_losses[epoch_idx]}
            val_metrics = {Metrics.DEFAULT_METRIC_LOSS.value : val_losses[epoch_idx] if epoch_idx < len(val_losses) else 0}
            
            for callback in filter(callable, callbacks):
                callback(epoch_idx, train_metrics, val_metrics)

    # ======================== BACKBONE CONTROL ========================

    def freeze_backbone(self):
        self._set_backbone_grad(requires_grad=False)
        logger.info("YOLO backbone frozen")

    def unfreeze_backbone(self):
        self._set_backbone_grad(requires_grad=True)
        logger.info("YOLO backbone unfrozen")

    def _set_backbone_grad(self, requires_grad: bool):
        for param in self.yolo_model.model.parameters():
            param.requires_grad = requires_grad

    # ======================== INFERENCE ========================

    def predict(self, images, confidence_threshold=ModelConstants.DEFAULT_CONFIDENCE_THRESHOLD) -> dict:
        self.eval()
        
        Results=ModelConstants.Results

        has_additional_layers = len(self.features) > 0
        
        if has_additional_layers:
            tensor_input = self._to_tensor(images)
            backbone_output = self.yolo_model.model(tensor_input)
            
            for layer in self.features:
                backbone_output = layer(backbone_output)
            
            results = self.yolo_model.predict(backbone_output, conf=confidence_threshold, verbose=False)
        else:
            results = self.yolo_model(images, conf=confidence_threshold, verbose=False)

        first_result = results[0]
        boxes = first_result.boxes.xyxy.cpu().numpy().tolist()
        scores = first_result.boxes.conf.cpu().numpy().tolist()

        return {
            Results.BOUNDING_BOXES_KEY: boxes,
            Results.CONFIDENCE_SCORES_KEY: scores,
            Results.NODULES_COUNT: len(boxes),
        }

    def _to_tensor(self, images):
        if isinstance(images, torch.Tensor):
            tensor = images
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            return tensor.to(self.device)


        is_numpy = isinstance(images, np.ndarray)
        tensor = to_tensor(images) if is_numpy else to_tensor(np.array(images))
        return tensor.unsqueeze(0).to(self.device)