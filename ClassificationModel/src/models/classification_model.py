import sys
from pathlib import Path
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import pickle
from constants.classification.model_constants import ModelConstants
from constants.classification.datasets_constants import DatasetConstants
from ClassificationModel.src.utils.dataset_utils import load_dataset
from utils.notification_service import NtfyNotificationService
from ClassificationModel.src.data_processing.image_augmentation import ImageAugmentationPipeline,apply_augmentation
class CancerClassificationModel:
    def __init__(self, dataset, input_shape,model_name=ModelConstants.MODEL_NAME):
        self.dataset = dataset
        self.notifier = NtfyNotificationService(model_name=model_name)
        self.input_shape = input_shape
        self.num_classes = dataset[DatasetConstants.NUM_CLASSES_KEY]
        self.class_names = dataset[DatasetConstants.CLASS_NAMES_KEY]
        self.model = None
        self.__build_model()



    def _add_conv_block(self, filters):
        '''
        a method to build a conv block consists of:
        2 convolution layers with normalization after each one 
        and a max pooling layer
        '''
        self.model.add(layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            padding=ModelConstants.PADDING_SAME,
            activation=ModelConstants.RELU_ACTIVATION_FUNCTION
        ))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            padding=ModelConstants.PADDING_SAME,
            activation=ModelConstants.RELU_ACTIVATION_FUNCTION
        ))
        self.model.add(layers.BatchNormalization())
        
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    def _add_dense_block(self, units):
        """Add a dense layer with batch norm."""
        self.model.add(layers.Dense(
            units=units,
            activation=ModelConstants.RELU_ACTIVATION_FUNCTION
        ))
        self.model.add(layers.BatchNormalization())
    
    def __build_model(self):
        """Build and compile the CNN model."""
        self.model = Sequential([layers.Input(shape=self.input_shape)])
        

        # Convolutional blocks
        self._add_conv_block(filters=32)
        self._add_conv_block(filters=64)
        self._add_conv_block(filters=128)
        
        # Flatten and dense layers
        self.model.add(layers.Flatten())
        self.model.add(layers.Dropout(0.3))
        self._add_dense_block(units=256)
        self._add_dense_block(units=128)
        
        self.model.add(layers.Dense(
            units=self.num_classes,
            activation=ModelConstants.OUTPUT_ACTIVATION
        ))
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=ModelConstants.LOSS_CATEGORICAL_CROSSENTROPY,
            metrics=[
                ModelConstants.METRIC_ACCURACY,
                tf.keras.metrics.Precision(name=ModelConstants.METRIC_PRECISION),
                tf.keras.metrics.Recall(name=ModelConstants.METRIC_RECALL),
                tf.keras.metrics.AUC(name=ModelConstants.METRIC_AUC, multi_label=False)
            ]
        )


    def train_model(self, epochs=ModelConstants.EPOCHS,callbacks=None,augment_train=False,augmenter=None):
        """Train the model on the dataset."""
        train_dataset = self.dataset[DatasetConstants.TRAIN_SPLIT_NAME]
        val_dataset = self.dataset[DatasetConstants.VAL_SPLIT_NAME]

        if augment_train:
            try:
                train_dataset = apply_augmentation(train_dataset,augmenter)    
            except ValueError as e:
                print(f'Error : {e}')
        
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history
    
    def evaluate_model(self,present_metrics=False,send_message=False):
        test_dataset = self.dataset[DatasetConstants.TEST_SPLIT_NAME]
        
        results = self.model.evaluate(test_dataset,return_dict=True, verbose=1)
        
        if present_metrics:
            for metric,value in results.items():
                print(f'{metric}: {value:.3f}')
        
        if send_message:
            metrics=NtfyNotificationService.format_metrics_msg(results)
            self.notifier.send_evaluation_results(metrics)

        return results
    
    def predict(self, images):
        predictions = self.model.predict(images)
        confidences = np.max(predictions,axis=1)
        predicted_indices = np.argmax(predictions, axis=1)
        predicted_class_names = [self.class_names[idx] for idx in predicted_indices]
        return [
            {
                ModelConstants.CANCER_TYPE_RESULT_KEY: cancer_type,
                ModelConstants.CONFIDENCE_KEY: confidence
            }
            for cancer_type, confidence in zip(predicted_class_names, confidences)
        ]


