
import requests
from datetime import datetime
from enum import StrEnum
from common.constants.notification import (
    NotificationPriority,
    NotificationTags,
    NotificationFields,
    NotificationHeaders,
    BASE_URL
)
from .constants import StatusCode


class NtfyNotificationService:
    def __init__(self,
                 model_name,
                 topic_name: str = NotificationFields.TOPIC_NAME,
                 base_url: str = BASE_URL):
        
        self.model_name = model_name
        self.topic_name = topic_name
        self.base_url = base_url
        self.url = f"{base_url}/{topic_name}"

    @staticmethod
    def format_metrics_msg(metrics):
        return "\n".join(f"{metric}: {value:.4f}" for metric, value in sorted(metrics.items()))

    def send_message(self, msg, title=None,
                      priority=NotificationPriority.DEFAULT,
                      tags: list[StrEnum] = None):
        headers = {
            NotificationHeaders.PRIORITY: priority,
            NotificationHeaders.TITLE: title or NotificationFields.DEFAULT_TITLE,
            NotificationHeaders.TAGS: ",".join(tags) if tags else "",
        }
        response = requests.post(
            self.url,
            data=msg.encode('utf-8'),
            headers=headers
        )

        return response.status_code == StatusCode.OK

    def send_training_start_message(self, total_epochs=None):
        epoch_info = f"\nTotal Epochs: {total_epochs}" if total_epochs else ""

        return self.send_message(
            msg=f"Model: {self.model_name}\nStarted at {datetime.now().strftime('%H:%M:%S')}{epoch_info}",
            title=NotificationFields.TRAINING_STARTED_TITLE,
            priority=NotificationPriority.DEFAULT,
            tags=[NotificationTags.START, NotificationTags.TRAINING]
        )

    def send_training_end_message(self, metrics, duration):
        return self.send_message(
            msg=f"Model: {self.model_name}\nTraining Duration: {duration:.1f} min\n\n{metrics}",
            title=NotificationFields.TRAINING_COMPLETED_TITLE,
            priority=NotificationPriority.HIGH,
            tags=[NotificationTags.COMPLETE, NotificationTags.SUCCESS]
        )

    def send_epoch_update(self, epoch, total_epochs, metrics):
        """Send notification for epoch completion"""
        return self.send_message(
            msg=f"Epoch {epoch}/{total_epochs}\n\n{metrics}",
            title=f"📊 Epoch {epoch} Complete",
            priority=NotificationPriority.LOW,
            tags=[NotificationTags.INFO]
        )

    def send_evaluation_results(self, metrics):
        return self.send_message(
            msg=metrics,
            title=NotificationFields.EVAL_RESULTS_TITLE,
            priority=NotificationPriority.HIGH,
            tags=[NotificationTags.COMPLETE, NotificationTags.SUCCESS]
        )
