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
from utils.base_cnn_model import BaseCNNModel
class CancerClassificationModel(BaseCNNModel):
    def __init__(self, dataset, input_shape, model_name=ModelConstants.MODEL_NAME, checkpoint_path=None):
        super().__init__(input_shape=input_shape, model_name=model_name)
        
        self.dataset = dataset
        self.num_classes = dataset[DatasetConstants.NUM_CLASSES_KEY]
        self.class_names = dataset[DatasetConstants.CLASS_NAMES_KEY]
        
        try:
            if checkpoint_path:
                self.load_checkpoint(checkpoint_path)
            else:
                self._build_model()
        except FileNotFoundError as e:
            print(f'Error : {e}')
            self._build_model()
    
    def _build_model(self):
        """Builds the specific architecture for Cancer Classification."""
        normalization_layer = tf.keras.layers.Normalization(axis=None)
        
        # Adapt normalization layer
        train_images = self.dataset[DatasetConstants.TRAIN_SPLIT_NAME].map(lambda x, y: x)
        normalization_layer.adapt(train_images)

        self.model = Sequential([
            layers.Input(shape=self.input_shape),
            normalization_layer
        ])
        
        # Using helper methods from BaseCNNModel
        self._add_conv_block(filters=32)
        self._add_conv_block(filters=64)
        self._add_conv_block(filters=128)
        
        self.model.add(layers.Flatten())
        self.model.add(layers.Dropout(0.3))
        
        self._add_dense_block(units=256)
        self._add_dense_block(units=128)
        
        self.model.add(layers.Dense(
            units=self.num_classes,
            activation=ModelConstants.OUTPUT_ACTIVATION
        ))
        
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


    def train_model(self, epochs=ModelConstants.EPOCHS, callbacks=None, augment_train=False, augmenter=None, class_weight=None):
        train_dataset = self.dataset[DatasetConstants.TRAIN_SPLIT_NAME]
        val_dataset = self.dataset[DatasetConstants.VAL_SPLIT_NAME]

        if augment_train and augmenter:
            try:
                train_dataset = apply_augmentation(train_dataset, augmenter)    
            except ValueError as e:
                print(f'Error applying augmentation: {e}')
        
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weight
        )
        return history
    
    def predict(self, images):
        predictions = self.model.predict(images)
        confidences = np.max(predictions, axis=1)
        predicted_indices = np.argmax(predictions, axis=1)
        predicted_class_names = [self.class_names[idx] for idx in predicted_indices]
        
        return [
            {
                ModelConstants.CANCER_TYPE_RESULT_KEY: cancer_type,
                ModelConstants.CONFIDENCE_KEY: confidence
            }
            for cancer_type, confidence in zip(predicted_class_names, confidences)
        ]
    
    def evaluate_model(self, present_metrics=False, send_message=False, save_confusion_matrix=True):
        """Specific evaluation logic for classification."""
        test_dataset = self.dataset[DatasetConstants.TEST_SPLIT_NAME]
        results = self.model.evaluate(test_dataset, return_dict=True, verbose=1)
        
        if present_metrics:
            for metric, value in results.items():
                print(f'{metric.upper()}: {value:.3f}')
        
        if save_confusion_matrix:
            self._save_confusion_matrix(test_dataset)
        
        if send_message:
            metrics_msg = NtfyNotificationService.format_metrics_msg(results)
            self.notifier.send_evaluation_results(metrics_msg)

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
    

