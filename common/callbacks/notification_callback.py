# notification_callback.py

import time
from tensorflow.keras.callbacks import Callback
from common.notification_service import NtfyNotificationService
import lightning as L
import torch


class NotificationCallback(Callback):
    def __init__(self,
                 notifier,
                 notify_on_epoch=False,
                 notify_every_n_epochs=10,
                 metrics_to_track=None):
        super().__init__()
        self.notifier = notifier
        self.notify_on_epoch = notify_on_epoch
        self.notify_every_n_epochs = notify_every_n_epochs
        self.metrics_to_track = metrics_to_track
        self.start_time = None
        self.total_epochs = None

    def _extract_metrics(self, logs):
        """Extract specified metrics from logs, or all if none specified"""
        metrics_to_use = self.metrics_to_track or logs.keys()
        return {k: logs[k] for k in metrics_to_use if k in logs}

    def on_train_begin(self, logs=None):
        """Send notification when training starts"""
        self.start_time = time.time()
        self.total_epochs = self.params.get('epochs', 'Unknown')
        self.notifier.send_training_start_message(total_epochs=self.total_epochs)

    def on_train_end(self, logs=None):
        """Send notification when training completes with final metrics"""
        duration = (time.time() - self.start_time) / 60
        metrics = NtfyNotificationService.format_metrics_msg(self._extract_metrics(logs))
        self.notifier.send_training_end_message(duration=duration, metrics=metrics)

    def on_epoch_end(self, epoch, logs=None):
        """Optionally send notification after each epoch"""
        should_notify = self.notify_on_epoch and (epoch + 1) % self.notify_every_n_epochs == 0
        metrics = NtfyNotificationService.format_metrics_msg(self._extract_metrics(logs))
        should_notify and self.notifier.send_epoch_update(
            epoch=epoch + 1,
            total_epochs=self.total_epochs,
            metrics=self._format_metrics_msg(metrics)
        )


class NtfyCallback(L.Callback):
    """Lightning callback for ntfy.sh training notifications."""

    def __init__(self, model_name, notify_every_n_epochs: int = 10, notify_on_epoch=False):
        super().__init__()
        self.notifier = NtfyNotificationService(model_name=model_name)
        self.notify_interval = notify_every_n_epochs
        self.notify_on_epoch = notify_on_epoch
        self.start_time = None

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        self.notifier.send_training_start_message()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1

        if self.notify_on_epoch and epoch % self.notify_interval == 0:
            metrics = self._extract_metrics(trainer)
            msg = NtfyNotificationService.format_metrics_msg(metrics)
            self.notifier.send_epoch_update(epoch=epoch, metrics=msg)

    def on_train_end(self, trainer, pl_module):
        duration = (time.time() - self.start_time) / 60
        metrics = self._extract_metrics(trainer)
        msg = NtfyNotificationService.format_metrics_msg(metrics)

        self.notifier.send_training_end_message(msg, duration)

    def on_test_end(self, trainer, pl_module):
        metrics = self._extract_metrics(trainer)
        clean_metrics = {k.replace("test_", ""): v for k, v in metrics.items()}
        msg = NtfyNotificationService.format_metrics_msg(clean_metrics)
        self.notifier.send_evaluation_results(msg)

    def _extract_metrics(self, trainer):
        return {
            k: v.item() if torch.is_tensor(v) else v
            for k, v in trainer.callback_metrics.items()
        }
