from abc import ABC, abstractmethod
from pathlib import Path
import logging
from utils.notification_service import NtfyNotificationService

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """
    Abstract base class for all models in the system.
    Handles common tasks like notifications and basic interface definition.
    """
    def __init__(self, model_name):
        self.model_name = model_name
        self.notifier = NtfyNotificationService(model_name=model_name)
        self.model = None

    @abstractmethod
    def load_checkpoint(self, checkpoint_path):
        """Loads the model weights/structure from a path."""
        pass

    @abstractmethod
    def train_model(self, *args, **kwargs):
        """Trains the model."""
        pass

    @abstractmethod
    def predict(self, input_data):
        """Performs prediction process."""
        pass
    
    @abstractmethod
    def _build_model(self):
        """Initializes the model structure if no checkpoint is found."""
        pass