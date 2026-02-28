import tensorflow as tf
from tensorflow.keras import layers
from pathlib import Path
from common.base_model import BaseModel
from common.constants import Activation
import logging

logger = logging.getLogger(__name__)


class BaseCNNModel(BaseModel):
    """
    Base class for Convolutional Neural Networks.
    Provides default implementations for Keras-based models.
    """

    def __init__(self, input_shape, model_name):
        super().__init__(model_name=model_name)
        self.input_shape = input_shape

    def load_checkpoint(self, checkpoint_path):
        """
        default implementation for checkpoint loading
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        try:
            self.model = tf.keras.models.load_model(checkpoint_path)
            logger.debug("Model loaded successfully")
        except Exception as e:
            logger.debug(f"Failed to load Keras model: {e}")
            raise e

    def _add_conv_block(self, filters, kernel_size=(3, 3), padding='same', activation=Activation.RELU):
        """Helper to add a standard convolution block (Conv -> BN -> MaxPool)."""
        self.model.add(layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            activation=activation
        ))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    def _add_dense_block(self, units, activation=Activation.RELU):
        self.model.add(layers.Dense(
            units=units,
            activation=activation
        ))
        self.model.add(layers.BatchNormalization())

    def evaluate_model(self, test_dataset, present_metrics=False, send_message=False, save_confusion_matrix=True):
        """Helper method for method evaluation, with the ability to present the results, and send a notification"""
        results = self.model.evaluate(test_dataset, return_dict=True, verbose=1)
        if present_metrics:
            for metric, value in results.items():
                print(f'{metric.upper()}: {value:.3f}')
        if send_message:
            from common.notification_service import NtfyNotificationService
            metrics_msg = NtfyNotificationService.format_metrics_msg(results)
            self.notifier.send_evaluation_results(metrics_msg)

        return results
