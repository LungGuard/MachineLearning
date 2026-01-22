import tensorflow as tf
from tensorflow.keras import layers
from pathlib import Path
from base_model import BaseModel

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
        Default implementation for loading Keras models.
        Child classes (like YOLO) should override this if they don't use Keras.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        try:
            self.model = tf.keras.models.load_model(checkpoint_path)
            print(f"✓ Model loaded successfully")
            if hasattr(self.model, 'input_shape'):
                 print(f"  Input shape: {self.model.input_shape}")
        except Exception as e:
            print(f"Failed to load Keras model: {e}")
            raise e

    def _add_conv_block(self, filters, kernel_size=(3, 3), padding='same', activation='relu'):
        """Helper to add a standard convolution block (Conv -> BN -> Pool)."""
        self.model.add(layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            activation=activation
        ))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    def _add_dense_block(self, units, activation='relu'):
        """Helper to add a dense block (Dense -> BN)."""
        self.model.add(layers.Dense(
            units=units,
            activation=activation
        ))
        self.model.add(layers.BatchNormalization())