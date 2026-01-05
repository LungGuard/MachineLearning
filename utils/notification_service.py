import requests
from datetime import datetime
from constants.notification_fields import NotificationPriority,NotificationTags,NotificationFields,NotificationHeaders


class NtfyNotificationService:
    def __init__(self, topic_name: str, base_url: str = "https://ntfy.sh"):
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


# Module-level singleton instance
_notification_service_instance = None

def get_notification_service(topic_name: str = NotificationFields.TOPIC_NAME, base_url: str = "https://ntfy.sh") -> NtfyNotificationService:
    """Get the global notification service instance"""
    global _notification_service_instance
    
    if _notification_service_instance is None:
        if topic_name is None:
            raise ValueError("topic_name is required for first initialization")
        _notification_service_instance = NtfyNotificationService(topic_name, base_url)
    elif topic_name and _notification_service_instance.topic_name != topic_name:
        # Create new instance if topic changed
        _notification_service_instance = NtfyNotificationService(topic_name, base_url)
    
    return _notification_service_instance

    