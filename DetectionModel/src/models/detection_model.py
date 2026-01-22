from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from constants.detection.model_constants import DetectionModelConstants
from utils.notification_service import NtfyNotificationService
from datetime import datetime
from utils.base_cnn_model import BaseCNNModel
import keras_cv as kcv

logger = logging.getLogger(__name__)

import tensorflow as tf
import keras_cv
from tensorflow.keras import layers, models
from pathlib import Path
import logging
from constants.detection.model_constants import DetectionModelConstants
from utils.base_cnn_model import BaseCNNModel

logger = logging.getLogger(__name__)

class NodulesDetectionModel(BaseCNNModel):
    def __init__(self, checkpoint_path=None, input_shape=(640, 640, 3)):
        # Initialize Base with generic params
        super().__init__(input_shape=input_shape, model_name=DetectionModelConstants.MODEL_NAME)
        
        self.num_classes = 1 # Assuming nodule vs background
        
        try:
            if checkpoint_path:
                self.load_checkpoint(checkpoint_path)
            else:
                self._build_model()
        except Exception as e:
            logger.error(f'Initialization Error: {e}')
            # Fallback to building a fresh model
            self._build_model()

    def _build_model(self):
        """
        Builds a Keras-native YOLOv8 model.
        Allows for true Transfer Learning (adding layers, freezing backbone).
        """        
        backbone = keras_cv.models.YOLOV8Backbone.from_preset(
            "yolo_v8_m_backbone_coco"  # 'n', 's', 'm', 'l', 'x' variants available
        )
        

        backbone.trainable = False 
        
        # 3. Create the Detector
        # keras_cv provides a ready-to-use YOLOV8Detector class that wraps the backbone
        # and adds the detection head.
        self.model = keras_cv.models.YOLOV8Detector(
            backbone=backbone,
            num_classes=self.num_classes,
            bounding_box_format="xywh", # Or 'xyxy', 'rel_yxyx' etc.
            # fpn_depth=2  # You can customize the Feature Pyramid Network depth here
        )
        
        # 4. Customizing/Adding Layers (The "Transfer Learning" part)
        # If you wanted to add custom classification layers *on top* of the features
        # instead of standard detection, you would access backbone.output
        # and build a functional API model:
        #
        # inputs = layers.Input(shape=self.input_shape)
        # x = backbone(inputs)
        # x = layers.GlobalAveragePooling2D()(x) # Process features
        # x = layers.Dense(128, activation='relu')(x) # YOUR CUSTOM LAYER
        # output = layers.Dense(1, activation='sigmoid')(x)
        # self.model = models.Model(inputs=inputs, outputs=output)

        # Compile with specific YOLO losses
        # Note: YOLO requires a complex box loss + classification loss combo.
        # KerasCV handles this internally if you use their Detector class.
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            classification_loss='binary_crossentropy',
            box_loss='ciou' 
        )

    def load_checkpoint(self, checkpoint_path):
        """
        Load Keras weights (.h5 or .keras format), NOT .pt files.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading Keras checkpoint from: {checkpoint_path}")
        # Standard Keras loading
        self.model.load_weights(checkpoint_path)
        print(f"✓ Model loaded successfully")

    def train_model(self, train_data, val_data=None, epochs=100, callbacks=None):
        """
        Now we can use standard .fit() because it's a real Keras model!
        """
        # Note: train_data must be a tf.data.Dataset dictionary with keys 
        # 'images' and 'bounding_boxes' formatted for KerasCV.
        
        print("Starting training with KerasCV YOLO...")
        return self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks
        )

    def predict(self, images):
        return self.model.predict(images)