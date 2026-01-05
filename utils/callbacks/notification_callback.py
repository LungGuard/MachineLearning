from constants.notification_fields import (NotificationFields,
                                        NotificationHeaders,
                                        NotificationPriority,
                                        NotificationTags
)
import time
from keras.callbacks import Callback
from utils.notification_service import NtfyNotificationService


class NotificationCallback(Callback):
    def __init__(self,
                 model_name: str,
                 notify_on_epoch = False,
                 notify_every_n_epochs = 10,
                 metrics_to_track=None):
        super().__init__()
        self.notifier = NtfyNotificationService(model_name=model_name)
        self.notify_on_epoch = notify_on_epoch
        self.notify_every_n_epochs = notify_every_n_epochs
        self.metrics_to_track = metrics_to_track
        self.start_time = None
    
    def _extract_metrics(self, logs):
        """Extract specified metrics from logs, or all if none specified"""
        metrics_to_use = self.metrics_to_track or logs.keys()
        return {k: logs[k] for k in metrics_to_use if k in logs} 
    
    def _format_metrics_msg(self, logs):
        """Format metrics to display them in a message"""
        metrics = self._extract_metrics(logs)
        return "\n".join(f"{metric}: {value:.4f}" for metric, value in metrics.items())
    
    def on_train_begin(self, logs=None):
        """Send notification when training starts"""
        self.start_time = time.time()
        self.notifier.send_training_start_message()

    def on_train_end(self, logs=None):
        """Send notification when training completes with final metrics"""
        duration = (time.time() - self.start_time) / 60
        metrics = self._format_metrics_msg(logs)
        self.notifier.send_training_end_message(duration=duration, metrics=metrics)
    
    def on_epoch_end(self, epoch, logs=None):
        """Optionally send notification after each epoch"""
        if self.notify_on_epoch and (epoch + 1) % self.notify_every_n_epochs == 0:
            metrics = self._format_metrics_msg(logs)
            self.notifier.send_epoch_update(epoch=epoch + 1, metrics=metrics)
