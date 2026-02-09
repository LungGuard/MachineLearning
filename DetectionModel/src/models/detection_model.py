"""
YOLO-based nodule detection model for LungGuard Stage 1.

Two training modes:
    1. YOLO-native (via dataset YAML) 
    2. Custom training (via DataLoader,using lightning)

Mode 1 bypasses the custom training loop entirely (YOLO has its own).
Mode 2 is using lighting to leverage its benefits benefits (auto device, checkpoints, logging).
"""

from pathlib import Path
import torch
import pandas as pd
import logging

from constants.detection.model_constants import DetectionModelConstants, YoloVariant
from utils.base_pt_cnn_model import BaseCNNModel
from ultralytics import YOLO
import lightning as L

logger = logging.getLogger(__name__)


class NodulesDetectionModel(BaseCNNModel):
    """
    Usage (YOLO-native):
        model = NodulesDetectionModel()
        model.set_dataset_config("data.yaml")
        history = model.train_yolo_native(epochs=100)

    Usage (Lightning):
        model = NodulesDetectionModel()
        trainer = L.Trainer(max_epochs=100, callbacks=[...])
        trainer.fit(model, train_loader, val_loader)
    """

    def __init__(self, checkpoint_path=None, input_shape=(3, 640, 640),
                 learning_rate=None):
        super().__init__(
            input_shape=input_shape,
            model_name=DetectionModelConstants.MODEL_NAME,
            learning_rate=learning_rate or DetectionModelConstants.LEARNING_RATE,
        )

        self.num_classes = 1
        self.yolo_model = None
        self.data_yaml_path = None

        self.initialize_from_checkpoint_or_build(
            checkpoint_path or "", freeze_params=True
        )

    def _build_model(self, freeze_params=True, additional_layers=None):
        """A method to build the yolo as the backbone and add additional layers if provided"""
        self.yolo_model = YOLO(YoloVariant.YOLO_LARGE.preset)

        if freeze_params:
            self.freeze_backbone()

        if additional_layers:
            self.features.extend(additional_layers)

        extra = f" + {len(additional_layers)} custom layers" if additional_layers else ""
        logger.info(f"Built: YOLO {YoloVariant.YOLO_LARGE.preset}{extra}")
        return self

    def forward(self, x):
        """Forward through YOLO backbone, then any additional layers."""
        out = self.yolo_model.model(x)

        for layer in self.features:
            out = layer(out)

        return out

    def forward_features(self, x):
        """
        Forward through YOLO backbone ONLY (no detection head).
        Returns raw feature maps — use this if your additional layers
        need standard (B, C, H, W) tensors.

        Useful for building custom heads on top of YOLO features.
        """
        return self.yolo_model.model.model[:10](x) 

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        training step for custom DataLoader mode.
        Auto-handles zero_grad, backward, optimizer step.
        """
        images, labels = batch
        raw_output = self.forward(images)
        loss = self.yolo_model.model.loss(raw_output, labels)

        # YOLO loss may return non-tensor in some edge cases
        if not torch.is_tensor(loss):
            loss = torch.tensor(loss, device=self.device, requires_grad=True)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Lightning validation step."""
        images, labels = batch
        raw_output = self.forward(images)
        loss = self.yolo_model.model.loss(raw_output, labels)

        if not torch.is_tensor(loss):
            loss = torch.tensor(loss, device=self.device)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss


    def set_dataset_config(self, yaml_path: str):
        """Set and validate YOLO dataset YAML path."""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset config not found: {yaml_path}")

        self.data_yaml_path = path
        logger.info(f"Dataset config: {path}")
        return self

    def train_yolo_native(self, epochs=100, learning_rate=None,
                          data_yaml=None, callbacks=None) -> dict:
        """
        Train using YOLO's built-in trainer (bypasses Lightning).

        Use this when you have a YOLO-format dataset YAML.
        For custom DataLoaders, use Lightning's Trainer.fit() instead.
        """
        config = data_yaml or self.data_yaml_path
        if not config:
            raise ValueError("No dataset config. Call set_dataset_config() first.")

        logger.info(f"YOLO-native training | config: {config}")

        args = {
            "data": str(config),
            "epochs": epochs,
            "imgsz": self.width,
            "batch": DetectionModelConstants.BATCH_SIZE,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "project": "checkpoints",
            "name": self.model_name,
            "exist_ok": True,
            "verbose": True,
        }

        if learning_rate:
            args["lr0"] = learning_rate

        results = self.yolo_model.train(**args)
        logger.info("YOLO-native training completed")

        history = self._parse_yolo_results(results)
        self._replay_callbacks(callbacks, history)
        return history

    def _parse_yolo_results(self, yolo_results) -> dict:
        """Convert YOLO results.csv → standard history dict."""
        csv_path = Path(yolo_results.save_dir) / "results.csv"

        if not csv_path.exists():
            logger.warning("results.csv not found — empty history")
            return {"train_loss": [], "val_loss": []}

        df = pd.read_csv(csv_path)
        cols = set(df.columns)

        return {
            "train_loss": df["train/box_loss"].tolist() if "train/box_loss" in cols else [],
            "val_loss": df["val/box_loss"].tolist() if "val/box_loss" in cols else [],
            "train_obj_loss": df["train/obj_loss"].tolist() if "train/obj_loss" in cols else [],
            "val_obj_loss": df["val/obj_loss"].tolist() if "val/obj_loss" in cols else [],
            "all_metrics": df.to_dict("list"),
        }

    def _replay_callbacks(self, callbacks, history):
        """Replay callbacks epoch-by-epoch from saved YOLO history."""
        if not callbacks:
            return

        for epoch in range(len(history.get("train_loss", []))):
            train_m = {"loss": history["train_loss"][epoch]}
            val_loss = history["val_loss"][epoch] if history["val_loss"] else 0.0
            val_m = {"loss": val_loss}

            for cb in callbacks:
                cb(epoch, train_m, val_m)

    # ======================== INFERENCE ========================

    def predict(self, images, confidence_threshold=0.25) -> dict:
        """
        Detect nodules with confidence filtering.

        """
        self.eval()
        tensor = self.ensure_batch_dim(images).to(self.device)

        with torch.no_grad():
            raw = self.forward(tensor)

        boxes, scores = self._filter_detections(raw, confidence_threshold)

        return {
            "bounding_boxes": boxes,
            "confidence_scores": scores,
            "num_nodules": len(boxes),
        }

    def _filter_detections(self, yolo_output, threshold: float):
        """Extract boxes + scores above confidence threshold."""
        result = yolo_output[0] if isinstance(yolo_output, (list, tuple)) else yolo_output

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        mask = scores >= threshold
        return boxes[mask].tolist(), scores[mask].tolist()


    def freeze_backbone(self):
        """Freeze YOLO backbone (for transfer learning)."""
        for p in self.yolo_model.model.parameters():
            p.requires_grad = False
        logger.info("YOLO backbone frozen")
        return self

    def unfreeze_backbone(self):
        """Unfreeze YOLO backbone (for fine-tuning)."""
        for p in self.yolo_model.model.parameters():
            p.requires_grad = True
        logger.info("YOLO backbone unfrozen")
        return self