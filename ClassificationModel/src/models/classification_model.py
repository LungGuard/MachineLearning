import sys
from pathlib import Path
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime
from constants.classification.model_constants import ModelConstants
from constants.classification.datasets_constants import DatasetConstants
from ClassificationModel.src.utils.dataset_utils import load_dataset
from utils.notification_service import NtfyNotificationService
from ClassificationModel.src.data_processing.image_augmentation import ImageAugmentationPipeline,apply_augmentation
class CancerClassificationModel:
    def __init__(self, dataset, input_shape, model_name=ModelConstants.MODEL_NAME, checkpoint_path=None):

        self.dataset = dataset
        self.notifier = NtfyNotificationService(model_name=model_name)
        self.input_shape = input_shape
        self.num_classes = dataset[DatasetConstants.NUM_CLASSES_KEY]
        self.class_names = dataset[DatasetConstants.CLASS_NAMES_KEY]
        self.model = None
        
        try:
            self.load_checkpoint(checkpoint_path)
        except FileNotFoundError as e:
            print(f'Error : {e}')
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
    
    def load_checkpoint(self, checkpoint_path):
        """Load a saved model checkpoint.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        self.model = tf.keras.models.load_model(checkpoint_path)
        print(f"✓ Model loaded successfully")
        print(f"  Input shape: {self.model.input_shape}")
        print(f"  Output classes: {self.model.output_shape[-1]}")
    
    def __build_model(self):
        """Build and compile the model."""

        normalization_layer= tf.keras.layers.Normalization(axis=None)
        
        train_images=self.dataset[DatasetConstants.TRAIN_SPLIT_NAME].map(lambda x,y:x) 
        normalization_layer.adapt(train_images)

        self.model = Sequential([
            layers.Input(shape=self.input_shape),
            normalization_layer
            ])
        

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


    def train_model(self, epochs=ModelConstants.EPOCHS,callbacks=None,augment_train=False,augmenter=None,class_weight=None):
        """Train the model on the dataset.
        
        Args:
            epochs: Number of training epochs
            callbacks: List of Keras callbacks
            augment_train: Whether to apply augmentation to training data
            augmenter: ImageAugmentationPipeline instance
            class_weight: Dictionary mapping class indices to weights for handling imbalance
        """
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
            callbacks=callbacks,
            class_weight=class_weight
        )
        
        return history
    
    def evaluate_model(self, present_metrics=False, send_message=False, save_confusion_matrix=True):
        """Evaluate the model on test data and optionally save confusion matrix.
        
        Args:
            present_metrics: Whether to print metrics
            send_message: Whether to send notification
            save_confusion_matrix: Whether to save confusion matrix plot
        
        Returns:
            Dictionary of evaluation metrics
        """
        test_dataset = self.dataset[DatasetConstants.TEST_SPLIT_NAME]
        
        results = self.model.evaluate(test_dataset, return_dict=True, verbose=1)
        
        if present_metrics:
            for metric, value in results.items():
                print(f'{metric}: {value:.3f}')
        
        if save_confusion_matrix:
            self._save_confusion_matrix(test_dataset)
        
        if send_message:
            metrics = NtfyNotificationService.format_metrics_msg(results)
            self.notifier.send_evaluation_results(metrics)

        return results
    
    def _save_confusion_matrix(self, test_dataset):
        """Generate and save confusion matrix visualization."""
        # Collect true labels and predictions
        y_true = []
        y_pred = []
        
        for images, labels in test_dataset:
            predictions = self.model.predict(images, verbose=0)
            y_true.extend(np.argmax(labels.numpy(), axis=1))
            y_pred.extend(np.argmax(predictions, axis=1))
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    ax=ax1, cbar_kws={'label': 'Count'})
        ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('True Label', fontsize=12)
        ax1.set_xlabel('Predicted Label', fontsize=12)
        
        # Plot 2: Percentages
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Greens',
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    ax=ax2, cbar_kws={'label': 'Percentage (%)'})
        ax2.set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('True Label', fontsize=12)
        ax2.set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        
        # Save to results folder
        results_dir = Path('results')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = results_dir / f'confusion_matrix_{timestamp}.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Confusion matrix saved to: {filepath}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names, digits=3))
    
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


