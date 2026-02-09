import torch
import torch.nn as nn
import lightning as L
from pathlib import Path
from torchmetrics import MetricCollection
from utils.notification_service import NtfyNotificationService
import logging

logger = logging.getLogger(__name__)


class BaseCNNModel(L.LightningModule):
    """
    Abstract base for all LungGuard CNN models.
    """

    def __init__(self, input_shape: tuple, model_name: str, learning_rate: float = 1e-3):
        super().__init__()

        self.save_hyperparameters()  # auto-saves init args for checkpoint reload

        self.model_name = model_name
        self.channels, self.height, self.width = input_shape
        self.learning_rate = learning_rate

        self.features = nn.ModuleList()
        self.notifier = NtfyNotificationService(model_name=model_name)

    def _build_model(self, freeze_params=True, **kwargs):
        """Subclasses build their architecture here."""
        raise NotImplementedError

    def forward(self, x):
        """Subclasses define forward pass."""
        raise NotImplementedError

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Subclasses compute and return loss for one batch.
        """
        raise NotImplementedError


    def validation_step(self, batch, batch_idx):
        """
        Default validation step — subclasses can override for custom metrics.
        Calls training_step and logs with 'val_' prefix.
        """
        loss = self.training_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        """Default: Adam over trainable params. Override for schedulers etc."""
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
        )

    def initialize_from_checkpoint_or_build(self, checkpoint_path: str, freeze_params=True):
        """Try loading checkpoint; fall back to building from scratch."""
        if self._try_load_checkpoint(checkpoint_path):
            return self

        self._build_model(freeze_params=freeze_params)
        return self

    def _try_load_checkpoint(self, checkpoint_path: str) -> bool:
        """Attempt to load weights. Returns True on success."""
        if not checkpoint_path:
            return False

        path = Path(checkpoint_path)
        if not path.exists():
            return False

        try:
            self.load_checkpoint(path)
            return True
        except Exception as e:
            logger.error(f"Checkpoint load failed for {self.model_name}: {e}")
            return False


    def load_checkpoint(self, checkpoint_path: Path):
        """Load model weights from a state_dict file."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        logger.info(f"Loading weights: {path}")
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)

    def save_checkpoint_manual(self, filepath: Path):
        """
        Manual state_dict save (for YOLO-native training or export).
        For Lightning-managed training, use ModelCheckpoint callback instead.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), filepath)
        logger.info(f"Checkpoint saved: {filepath}")

    # ======================== EVALUATION ========================

    def evaluate_model(self, test_loader, metrics: dict = None,
                       print_results=False, send_notification=False) -> dict:
        """
        Run evaluation on test data.

        Note: For standard eval, prefer Lightning's Trainer.test().
        This method exists for custom metric bundles and notification integration.
        """
        results = self._compute_eval_metrics(test_loader, metrics)

        if print_results:
            self._print_metrics(results)

        if send_notification:
            self._notify_results(results)

        return results

    def _compute_eval_metrics(self, test_loader, metrics: dict = None) -> dict:
        """Evaluation loop with optional torchmetrics."""
        self.eval()

        collection = self._init_metric_collection(metrics)
        running_loss, n_samples = 0.0, 0

        with torch.no_grad():
            for batch in test_loader:
                loss = self.training_step(batch, 0) # Subclass training_step returns loss
                
                batch_size = batch[0].size(0) if isinstance(batch, (list, tuple)) else 1

                running_loss += loss.item() * batch_size
                n_samples += batch_size

                if collection:
                    data, target = batch[0].to(self.device), batch[1].to(self.device)
                    output = self(data)
                    collection.update(output, target)

        results = {"loss": running_loss / max(n_samples, 1)}

        if collection:
            computed = collection.compute()
            results.update({k: v.item() for k, v in computed.items()})
            collection.reset()

        return results

    def _init_metric_collection(self, metrics):
        """Wrap metrics into MetricCollection on correct device."""
        if not metrics:
            return None

        collection = MetricCollection(metrics) if isinstance(metrics, dict) else metrics
        collection = collection.to(self.device)
        collection.reset()
        return collection

    def _notify_results(self, results: dict):
        try:
            msg = NtfyNotificationService.format_metrics_msg(results)
            self.notifier.send_evaluation_results(msg)
        except Exception as e:
            logger.warning(f"Notification failed: {e}")

    @staticmethod
    def _print_metrics(results: dict):
        for name, value in results.items():
            print(f"{name.upper()}: {value:.3f}")


    def ensure_batch_dim(self, images: torch.Tensor) -> torch.Tensor:
        """Add batch dimension to a single image tensor."""
        return images.unsqueeze(0) if images.dim() == 3 else images

    def _get_conv_block(self, in_ch, out_ch, kernel_size=3, activation=None):
        from .pt_layers.Conv2D_block import Conv2DBlock
        return Conv2DBlock(in_ch, out_ch, kernel_size, activation or nn.ReLU())

    def _get_dense_block(self, in_features, out_features, activation=None):
        from .pt_layers.DenseBlock import DenseBlock
        return DenseBlock(in_features, out_features, activation or nn.ReLU())