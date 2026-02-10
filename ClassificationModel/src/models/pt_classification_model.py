import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from constants.classification.model_constants import ModelConstants
from constants.classification.datasets_constants import DatasetConstants
from utils.base_pt_cnn_model import BaseCNNModel
import torch
import torch.nn as nn


class PtCancerClassificationModel(BaseCNNModel):
    def __init__(self, input_shape, model_name,dataset,checkpoint_path=None):
        super().__init__(input_shape, model_name)
        self.dataset=dataset
        try:
            if checkpoint_path:
                self.load_checkpoint(checkpoint_path)
            else:
                self._build_model()
        except FileNotFoundError as e:
            print(f'Error : {e}')
            self._build_model()

    def _build_model(self,dropout_rate=0.3):
        self.features.append(self._get_conv_block(self.channels,32))
        self.features.append(self._get_conv_block(32,64))
        self.features.append(self._get_conv_block(64, 128))
        self.classifier=nn.Sequential(
            nn.Flatten(),

            nn.Dropout(dropout_rate),
            
            nn.LazyLinear(256),
            nn.BatchNorm1d(256),  
            nn.ReLU(),
            
            self._get_dense_block(256,128),
            
            nn.Linear(128, self.dataset.num_classes)
        )
    
    def forward(self,x):
        for layer in self.features:
            x = layer(x)
        
        x = self.classifier(x)
        
        return x
    
    def predict(self,images):
        
        self.eval()
        device=next(self.parameters()).device
        images = images.to(device)

        with torch.no_grad():
            logits = self(images)
            probabilities = torch.softmax(logits, dim=1)
            confidences, predicted_indices = torch.max(probabilities, 1)
            
            confidences = confidences.cpu().numpy()
            predicted_indices = predicted_indices.cpu().numpy()
            class_names = self.dataset.class_names
            
            return [
                {
                    ModelConstants.CANCER_TYPE_RESULT_KEY: class_names[idx],
                    ModelConstants.CONFIDENCE_KEY: float(conf)
                }
                for idx, conf in zip(predicted_indices, confidences)
            ]