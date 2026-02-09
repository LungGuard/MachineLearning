

import torch as pt
import torch.nn as nn
from pathlib import Path
from torchmetrics import MetricCollection
import logging
from .pt_layers.Conv2D_block import Conv2DBlock
from .pt_layers.DenseBlock import DenseBlock
from utils.notification_service import NtfyNotificationService

logger = logging.getLogger(__name__)

class BaseCNNModel(nn.Module):
    """
    Base class for Convolutional Neural Networks in PyTorch.
    """

    def __init__(self, input_shape, model_name):
        super(BaseCNNModel, self).__init__()
        self.model_name = model_name
        self.channels,self.height,self.width, = input_shape 
        self.notifier=NtfyNotificationService(model_name=model_name)
        self.features = nn.ModuleList()

    def load_checkpoint(self, checkpoint_path):
        """
        Implementation for PyTorch state_dict loading
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading weights from: {checkpoint_path}")
        try:
            state_dict = pt.load(checkpoint_path, map_location=pt.device('cpu'))
            self.load_state_dict(state_dict)
            logger.debug("Weights loaded successfully")
        except Exception as e:
            logger.debug(f"Failed to load PyTorch model: {e}")
            raise e

    def _get_conv_block(self, in_channels, out_channels, kernel_size=3, activation=nn.ReLU()):
        """Helper to create a standard convolution block."""
        return Conv2DBlock(in_channels, out_channels, kernel_size, activation)

    def _get_dense_block(self, in_features, out_features, activation=nn.ReLU()):
        """Helper to create a standard dense block."""
        return DenseBlock(in_features, out_features, activation)
    
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward()")
    
    def evaluate_model(self, test_loader, criterion, metrics=None, present_metrics=False, send_message=False):
        """
        Main evaluation method. Orchestrates calculation, presentation, and notification.
        """
        # 1. Responsibility: Calculate Metrics
        results = self._calculate_metrics(test_loader, criterion, metrics)

        # 2. Responsibility: Present Results
        if present_metrics:
            self._present_results(results)

        # 3. Responsibility: Send Notification
        if send_message:
            self._send_notification(results)

        return results

    def _calculate_metrics(self, test_loader, criterion, metrics):
        """
        Handles the PyTorch evaluation loop and metric computation.
        Returns a dictionary of results (e.g., {'loss': 0.5, 'accuracy': 0.9}).
        """
        self.eval()
        device = next(self.parameters()).device
        
        if metrics:
            if isinstance(metrics, dict):
                metrics = MetricCollection(metrics).to(device)
            else:
                metrics = metrics.to(device)
            metrics.reset()

        running_loss = 0.0
        total_samples = 0
        
        with pt.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                
                output = self(data)
                
                # Loss calculation
                loss = criterion(output, target)
                running_loss += loss.item() * data.size(0)
                
                # Metric updates
                if metrics:
                    metrics.update(output, target)
                
                total_samples += data.size(0)

        # Finalize calculations
        avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
        results = {'loss': avg_loss}
        
        if metrics:
            computed = metrics.compute()
            # Convert tensors to python floats and merge into results
            results.update({k: v.item() for k, v in computed.items()})
            metrics.reset()
            
        return results

    def _present_results(self, results):
        """
        Handles printing results to the console.
        """
        print(f"\n--- Evaluation Results: {self.model_name} ---")
        for metric, value in results.items():
            print(f'{metric.upper()}: {value:.3f}')

    def _send_notification(self, results):
        """
        Handles formatting and sending the notification via Ntfy.
        """
        
        try:
            metrics_msg = NtfyNotificationService.format_metrics_msg(results)
            self.notifier.send_evaluation_results(metrics_msg)
        except Exception as e:
            print(f"Warning: Failed to send notification: {e}")
