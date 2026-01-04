import requests
from datetime import datetime
from constants.notification_fields import NotificationPriority,NotificationTags,NotificationFields,NotificationHeaders


class NtfyNotificationService:

    def __init__(self, topic_name: str, base_url: str = "https://ntfy.sh"):
        self.topic_name = topic_name
        self.base_url = base_url
        self.url = f"{base_url}/{topic_name}"
    
    def send_message(self,msg,title,priority,tags):
        headers = {NotificationHeaders.PRIORITY_HEADER: priority.value}
        headers[NotificationHeaders.TITLE_HEADER] = title or NotificationFields.DEFAULT_TITLE
        headers[NotificationHeaders.TAGS_HEADER] = ",".join(tag.value if isinstance(tag, NotificationTags) else tag for tag in tags) if tags else ""
        
        response = requests.post(
            self.url,
            data=msg.encode('utf-8'),
            headers=headers
        )

        return response.status_code == 200
