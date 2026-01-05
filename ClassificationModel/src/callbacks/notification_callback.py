from constants.notification_fields import (NotificationFields,
                                        NotificationHeaders,
                                        NotificationPriority,
                                        NotificationTags
)
import time
from keras.callbacks import Callback
from utils.notification_service import get_notification_service

class NotificationCallback(Callback):
    def __init__(self,
                 notify_on_epoch = False,
                 notify_every_n_epochs = 10,
                 metrics_to_track=None):
        super().__init__()
        self.notification_service = get_notification_service()
        self.notify_on_epoch = notify_on_epoch
        self.notify_every_n_epochs = notify_every_n_epochs
        self.metrics_to_track=metrics_to_track
    
    def _extract_metrics(self,logs):
        metrics_to_use = self.metrics_to_track or logs.keys()
        return {k:logs[k] for k in metrics_to_use if k in logs} 
    
    def _format_train_end_notification_msg(self, logs):
        metrics = self._extract_metrics(logs)
        return "\n".join(f"{metric}: {value}" for metric, value in metrics.items())
    
    def on_train_end(self, logs=None):
        pass
