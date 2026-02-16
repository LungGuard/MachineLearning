import lightning as L
import torch
import torch.optim as optim
import logging
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.ops import non_max_suppression

logger = logging.getLogger(__name__)

class NodulesDetectionModel(L.LightningModule):
    def __init__(self, 
                 model_yaml_path: str = "yolov8l.yaml", 
                 pretrained_weights: str = "yolov8l.pt", 
                 num_classes: int = 1,
                 learning_rate: float = 1e-4):
        
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
        imgs, targets = batch
        preds = self.model(imgs)
        
        # חישוב ה-Loss. שימו לב: v8DetectionLoss מחזירה טנסור Loss ופריטים לפירוט
        loss, loss_items = self.loss_fn(preds, batch)
        
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/box_loss", loss_items[0])
        self.log("train/cls_loss", loss_items[1])
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        preds = self.model(imgs)
        loss, loss_items = self.loss_fn(preds, batch)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        return loss


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Main entry point for inference.
        Orchestrates detection, conversion to XYHW, and crop extraction.
        """
        stacked_slices_batch, _ = batch 
        
        # 1. חיזוי גולמי מהמודל
        raw_predictions = self.model(stacked_slices_batch)
        
        # 2. סינון תוצאות (NMS) - מחזיר [x1, y1, x2, y2, conf, cls]
        refined_detections = self._filter_detections(raw_predictions)
        
        # 3. עיבוד ה-Batch למבנה הסופי (XYHW + Crops)
        return self._format_batch_results(stacked_slices_batch, refined_detections, batch_idx)

    def _filter_detections(self, predictions):
        return non_max_suppression(predictions, conf_thres=0.25, iou_thres=0.45)

    def _format_batch_results(self, batch_images, detections_list, batch_idx):
        batch_results = []
        for i, detections in enumerate(detections_list):
            sample_data = self._process_single_sample(
                full_sandwich=batch_images[i],
                detections=detections,
                sample_idx=i,
                batch_idx=batch_idx
            )
            batch_results.append(sample_data)
        return batch_results

    def _process_single_sample(self, full_sandwich, detections, sample_idx, batch_idx):
        """
        מעבד דגימה בודדת: מחלץ סלייס אמצעי מלא למסווג וקופסאות XYHW לרגרסיה.
        """
        # --- CLASSIFIER INPUT: הסלייס האמצעי המלא (Channel 1) ---
        full_middle_slice = full_sandwich[1:2, :, :].cpu()
        
        detected_nodules = []
        
        for det in detections:
            # המרה מ-XYXY (פינות) ל-XYHW (מרכז, גובה, רוחב)
            xyhw_meta = self._convert_to_xyhw(det[:4])
            
            # חילוץ ה-Crop עבור הרגרסיה (3 ערוצים חתוכים)
            regression_crop = self._extract_regression_crop(full_sandwich, det[:4])
            
            detected_nodules.append({
                "x": xyhw_meta["x"],
                "y": xyhw_meta["y"],
                "h": xyhw_meta["h"],
                "w": xyhw_meta["w"],
                "confidence": det[4].item(),
                "regression_input": regression_crop 
            })

        return {
            "metadata": {"batch_idx": batch_idx, "sample_idx": sample_idx},
            "classifier_input": full_middle_slice,
            "nodules": detected_nodules
        }

    def _convert_to_xyhw(self, bbox_xyxy):
        """
        Converts bounding box from (x1, y1, x2, y2) to (x_center, y_center, height, width).
        """
        x1, y1, x2, y2 = bbox_xyxy
        
        width = x2 - x1
        height = y2 - y1
        x_center = x1 + (width / 2)
        y_center = y1 + (height / 2)
        
        return {
            "x": x_center.item(),
            "y": y_center.item(),
            "h": height.item(),
            "w": width.item()
        }

    def _extract_regression_crop(self, stacked_slices, bbox_xyxy):
        """
        גוזר את הגידול ב-3 ערוצים עבור מודל הרגרסיה.
        """
        x1, y1, x2, y2 = map(int, bbox_xyxy)
        _, img_h, img_w = stacked_slices.shape
        
        # הגנה על גבולות התמונה
        x1, x2 = max(0, x1), min(img_w, x2)
        y1, y2 = max(0, y1), min(img_h, y2)
        
        return stacked_slices[:, y1:y2, x1:x2].cpu()


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            }
        }