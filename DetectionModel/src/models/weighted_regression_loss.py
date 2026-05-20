import torch
import torch.nn as nn
from DetectionModel.constants.enums.features import Features

_FEATURE_WEIGHTS = {
    'diameter_mm':        1.0,
    'malignancy':         1.0,
    'spiculation':        1.0,
    'lobulation':         1.0,
    'subtlety':           1.0,
    'sphericity':         1.0,
    'margin':             1.0,
    'texture':            1.0,
    'calcification':      0.3,
    'internal_structure': 0.1,
}

def _build_weight_tensor(feature_names: list[str]) -> torch.Tensor:
    weights = [_FEATURE_WEIGHTS.get(f, 1.0) for f in feature_names]
    t = torch.tensor(weights, dtype=torch.float32)
    return t / t.sum() * len(t)


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        features = Features.getNoduleFeaturesVector()
        self.register_buffer('weights', _build_weight_tensor(features))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sq_err = (pred - target) ** 2
        return (sq_err * self.weights).mean()


class WeightedHuberLoss(nn.Module):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta
        features = Features.getNoduleFeaturesVector()
        self.register_buffer('weights', _build_weight_tensor(features))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err = pred - target
        abs_err = err.abs()
        huber = torch.where(
            abs_err <= self.delta,
            0.5 * err ** 2,
            self.delta * (abs_err - 0.5 * self.delta),
        )
        return (huber * self.weights).mean()
