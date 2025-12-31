import sys
from pathlib import Path
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers, layers
import pickle
from common.constants.model_constants import ModelConstants
from common.constants.datasets_constants import DatasetConstants
from data_processing.dataset_utils import load_dataset


class CancerClassificationModel:
    def __init__(self,dataset,input_shape):
        self.model = None
        self.dataset = dataset
        self.input_shape = input_shape
        self.num_classes=dataset[DatasetConstants.NUM_CLASSES_KEY]
        self.class_names=dataset[DatasetConstants.CLASS_NAMES_KEY]

    def _add_conv_block(self, filters, block_num, dropout_rate=0.25, l2_reg=0.001):
        """Add a convolutional block with 2 conv layers, batch norm, maxpool, and dropout."""
        block_prefix = ModelConstants.BLOCK_PREFIXES[block_num]
        maxpool_name = ModelConstants.MAXPOOL_NAMES[block_num]
        
        # First conv layer
        self.model.add(layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=ModelConstants.PADDING_SAME,
            activation=ModelConstants.RELU_ACTIVATION_FUNCTION,
            kernel_initializer=ModelConstants.KERNEL_INITIALIZER,
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f'{block_prefix}1'
        ))
        self.model.add(layers.BatchNormalization(name=f'{ModelConstants.BN_PREFIX}{block_num}_1'))
        
        # Second conv layer
        self.model.add(layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=ModelConstants.PADDING_SAME,
            activation=ModelConstants.RELU_ACTIVATION_FUNCTION,
            kernel_initializer=ModelConstants.KERNEL_INITIALIZER,
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f'{block_prefix}2'
        ))
        self.model.add(layers.BatchNormalization(name=f'{ModelConstants.BN_PREFIX}{block_num}_2'))
        
        # MaxPooling and Dropout
        self.model.add(layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            name=maxpool_name
        ))
        self.model.add(layers.Dropout(dropout_rate, name=f'{ModelConstants.DROPOUT_PREFIX}{block_num}'))
    
    def _add_dense_block(self, units, block_num, dropout_rate=0.5, l2_reg=0.001):
        """Add a dense block with batch norm and dropout."""
        self.model.add(layers.Dense(
            units=units,
            activation=ModelConstants.RELU_ACTIVATION_FUNCTION,
            kernel_initializer=ModelConstants.KERNEL_INITIALIZER,
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f'{ModelConstants.DENSE_PREFIX}{block_num}'
        ))
        self.model.add(layers.BatchNormalization(name=f'{ModelConstants.BN_DENSE_PREFIX}{block_num}'))
        self.model.add(layers.Dropout(dropout_rate, name=f'{ModelConstants.DROPOUT_DENSE_PREFIX}{block_num}'))
    
    def build_model(self):
        """Build the CNN model with 4 convolutional blocks and 2 dense layers."""
        self.model = Sequential([
            layers.Input(shape=self.input_shape)
        ])
        
        # First convolutional block
        self.model.add(layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=ModelConstants.PADDING_SAME,
            activation=ModelConstants.RELU_ACTIVATION_FUNCTION,
            kernel_initializer=ModelConstants.KERNEL_INITIALIZER,
            kernel_regularizer=regularizers.l2(0.001),
            name=f'{ModelConstants.BLOCK_PREFIXES[1]}1'
        ))

        self.model.add(layers.BatchNormalization(name=f'{ModelConstants.BN_PREFIX}1_1'))
        self.model.add(layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=ModelConstants.PADDING_SAME,
            activation=ModelConstants.RELU_ACTIVATION_FUNCTION,
            kernel_initializer=ModelConstants.KERNEL_INITIALIZER,
            kernel_regularizer=regularizers.l2(0.001),
            name=f'{ModelConstants.BLOCK_PREFIXES[1]}2'
        ))
        self.model.add(layers.BatchNormalization(name=f'{ModelConstants.BN_PREFIX}1_2'))
        
        self.model.add(layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            name=ModelConstants.MAXPOOL_NAMES[1]
        ))
        self.model.add(layers.Dropout(0.25, name=f'{ModelConstants.DROPOUT_PREFIX}1'))
        
        self._add_conv_block(filters=64, block_num=2)
        self._add_conv_block(filters=128, block_num=3)
        self._add_conv_block(filters=256, block_num=4)
        
        self.model.add(layers.Flatten(name=ModelConstants.FLATTEN_NAME))
        
        # Dense blocks
        self._add_dense_block(units=512, block_num=1)
        self._add_dense_block(units=256, block_num=2)
        
        # Output layer
        self.model.add(layers.Dense(
            units=self.num_classes,
            activation=ModelConstants.OUTPUT_ACTIVATION,
            kernel_initializer=ModelConstants.XAVIER_INITIALIZER,
            name=ModelConstants.OUTPUT_LAYER_NAME
        ))
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=ModelConstants.LOSS_CATEGORICAL_CROSSENTROPY,
            metrics=[
                ModelConstants.METRIC_ACCURACY,
                tf.keras.metrics.Precision(name=ModelConstants.METRIC_PRECISION),
                tf.keras.metrics.Recall(name=ModelConstants.METRIC_RECALL)
            ]
        )
    def get_model_summary(self) -> str:
        """Get detailed model architecture summary."""
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list)
    

    def train_model(self, epochs=ModelConstants.EPOCHS):
        """Train the model on the dataset."""
        train_dataset = self.dataset[DatasetConstants.TRAIN_SPLIT_NAME]
        val_dataset = self.dataset[DatasetConstants.VAL_SPLIT_NAME]
        
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs
        )
        
        return history
    
    def evaluate_model(self):
        test_dataset = self.dataset[DatasetConstants.TEST_SPLIT_NAME]
        
        results = self.model.evaluate(test_dataset, verbose=1)
        
        metrics = {
            ModelConstants.LOSS_METRIC: results[0],
            ModelConstants.METRIC_ACCURACY: results[1],
            ModelConstants.METRIC_PRECISION: results[2],
            ModelConstants.METRIC_RECALL: results[3]
        }
        
        return metrics


