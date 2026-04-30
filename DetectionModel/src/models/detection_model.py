import logging

import lightning as L
import torch
import torch.optim as optim
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.ops import non_max_suppression

from DetectionModel.constants.constants.detection_model import DetectionModelConstants
from DetectionModel.src.data_preprocessing.core.scan_protocols import (
    DetectedNodule,
    SliceDetectionResult,
)

logger = logging.getLogger(__name__)


class NodulesDetectionModel(L.LightningModule):
    def __init__(self,
                 model_yaml_path: str = DetectionModelConstants.DEFAULT_MODEL_YAML,
                 pretrained_weights: str = DetectionModelConstants.DEFAULT_PRETRAINED_WEIGHTS,
                 num_classes: int = DetectionModelConstants.NUM_CLASSES,
                 learning_rate: float = DetectionModelConstants.LEARNING_RATE):

        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.model = DetectionModel(cfg=model_yaml_path, nc=num_classes)

        if pretrained_weights:
            self._load_weights(pretrained_weights)

        self.loss_fn = v8DetectionLoss(self.model)

    def _load_weights(self, weights_path):
        try:
            ckpt = torch.load(weights_path, map_location='cpu')
            state_dict = ckpt['model'].float().state_dict() if 'model' in ckpt else ckpt
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"Successfully loaded weights from {weights_path}")
        except Exception as e:
            logger.warning(f"Failed to load weights: {e}. Training from scratch.")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        preds = self.model(imgs)

        loss, loss_items = self.loss_fn(preds, batch)

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/box_loss", loss_items[0])
        self.log("train/cls_loss", loss_items[1])
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, _ = batch
        preds = self.model(imgs)
        loss, _ = self.loss_fn(preds, batch)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> list[SliceDetectionResult]:
        stacked_slices_batch, _ = batch

        raw_predictions = self.model(stacked_slices_batch)
        refined_detections = self._apply_nms(raw_predictions)

        return self._format_batch_results(stacked_slices_batch, refined_detections, batch_idx)

    def _apply_nms(self, predictions):
        return non_max_suppression(
            predictions,
            conf_thres=DetectionModelConstants.NMS_CONF_THRESHOLD,
            iou_thres=DetectionModelConstants.NMS_IOU_THRESHOLD,
        )

    def _format_batch_results(self, batch_images, detections_list, batch_idx) -> list[SliceDetectionResult]:
        return [
            self._process_single_sample(
                full_sandwich=batch_images[i],
                detections=detections,
                sample_idx=i,
                batch_idx=batch_idx,
            )
            for i, detections in enumerate(detections_list)
        ]

    def _process_single_sample(self, full_sandwich, detections, sample_idx, batch_idx) -> SliceDetectionResult:
        classifier_input = full_sandwich[1:2, :, :].cpu()

        detected_nodules = [
            self._build_detected_nodule(full_sandwich, det)
            for det in detections
        ]

        return SliceDetectionResult(
            batch_idx=batch_idx,
            sample_idx=sample_idx,
            classifier_input=classifier_input,
            nodules=detected_nodules,
        )

    def _build_detected_nodule(self, full_sandwich, det) -> DetectedNodule:
        bbox_xyxy = det[:4]
        x_center, y_center, height, width = self._convert_xyxy_to_xyhw(bbox_xyxy)
        regression_crop = self._extract_regression_crop(full_sandwich, bbox_xyxy)

        return DetectedNodule(
            x=x_center,
            y=y_center,
            h=height,
            w=width,
            confidence=det[4].item(),
            regression_input=regression_crop,
        )

    @staticmethod
    def _convert_xyxy_to_xyhw(bbox_xyxy) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox_xyxy
        width = x2 - x1
        height = y2 - y1
        x_center = x1 + (width / 2)
        y_center = y1 + (height / 2)
        return x_center.item(), y_center.item(), height.item(), width.item()

    @staticmethod
    def _extract_regression_crop(stacked_slices, bbox_xyxy) -> torch.Tensor:
        x1, y1, x2, y2 = map(int, bbox_xyxy)
        _, img_h, img_w = stacked_slices.shape
        x1, x2 = max(0, x1), min(img_w, x2)
        y1, y2 = max(0, y1), min(img_h, y2)
        return stacked_slices[:, y1:y2, x1:x2].cpu()

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=DetectionModelConstants.OPTIMIZER_WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=DetectionModelConstants.LR_SCHEDULER_FACTOR,
            patience=DetectionModelConstants.LR_SCHEDULER_PATIENCE,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }
