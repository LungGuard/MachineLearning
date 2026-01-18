from enum import StrEnum,Enum


class NotificationPriority(StrEnum):
    MIN = "min"
    LOW = "low"
    DEFAULT = "default"
    HIGH = "high"
    URGENT = "urgent"

class NotificationTags(StrEnum):
    SUCCESS = "white_check_mark"
    ERROR = "x"
    WARNING = "warning"
    INFO = "information_source"
    TRAINING = "chart_with_upwards_trend"
    COMPLETE = "tada"
    START = "rocket"

class NotificationFields(StrEnum):
    DEFAULT_TITLE = "LungGuard Notification"
    TOPIC_NAME = "FinalsProjectNotifications"
    TRAINING_STARTED_TITLE = "Training Started!"
    TRAINING_COMPLETED_TITLE = "Training Completed!"
    EVAL_RESULTS_TITLE="Evaluation Results"

class NotificationHeaders(StrEnum):
    PRIORITY_HEADER = "Priority"
    TITLE_HEADER = "Title"
    TAGS_HEADER = "Tags"