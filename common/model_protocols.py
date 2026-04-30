
from typing import  Protocol, runtime_checkable
import torch
import numpy as np
from common.dto import CancerClass, NoduleFeatures

from DetectionModel.src.data_preprocessing.core.scan_protocols import (
    SliceDetectionResult,
)

@runtime_checkable
class DetectionModelProtocol(Protocol):
    def eval(self) -> None: ...
    def predict_step(
        self, batch: tuple, batch_idx: int, dataloader_idx: int = 0
    ) -> list[SliceDetectionResult]: ...


@runtime_checkable
class RegressionModelProtocol(Protocol):
    def predict_features(self, x: torch.Tensor) -> list[NoduleFeatures]: ...


@runtime_checkable
class ClassificationModelProtocol(Protocol):
    def predict(self, images: np.ndarray) -> list[CancerClass]: ...
