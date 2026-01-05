import requests
from datetime import datetime
from constants.notification_fields import NotificationPriority, NotificationTags, NotificationFields, NotificationHeaders


class NtfyNotificationService:
    def __init__(self,
                 model_name
                 ,topic_name: str = NotificationFields.TOPIC_NAME
                 , base_url: str = "https://ntfy.sh"):
        self.model_name=model_name
        self.topic_name = topic_name
        self.base_url = base_url
        self.url = f"{base_url}/{topic_name}"
    
    def send_message(self, msg, title=None, priority=NotificationPriority.DEFAULT, tags=None):
        headers = {NotificationHeaders.PRIORITY_HEADER: priority.value}
        headers[NotificationHeaders.TITLE_HEADER] = title or NotificationFields.DEFAULT_TITLE
        headers[NotificationHeaders.TAGS_HEADER] = ",".join(tag.value if isinstance(tag, NotificationTags) else tag for tag in tags) if tags else ""
        
        response = requests.post(
            self.url,
            data=msg.encode('utf-8'),
            headers=headers
        )

        return response.status_code == 200
    
    def send_training_start_message(self):
        return self.send_message(
            msg=f"Model: {self.model_name}\nStarted at {datetime.now().strftime('%H:%M:%S')}",
            title=NotificationFields.TRAINING_STARTED_TITLE,
            priority=NotificationPriority.DEFAULT,
            tags=[NotificationTags.START, NotificationTags.TRAINING]
        )
    
    def send_training_end_message(self,metrics,duration):
        return self.send_message(
            msg=f"Model : {self.model_name}\nRunning time: {duration:.1f}\n{metrics}",
            title=NotificationFields.TRAINING_COMPLETED_TITLE,
            priority=NotificationPriority.HIGH,
            tags=[NotificationTags.COMPLETE, NotificationTags.SUCCESS]
        )


