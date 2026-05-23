import torch
from pydantic import BaseModel

# Inverse of the StandardScaler fitted on the LIDC-IDRI training split.
# Features.getNoduleFeaturesVector() order (indices 0-9):
#   [0] DIAMETER_MM  [1] MALIGNANCY  [2] SPICULATION  [3] LOBULATION
#   [4] SUBTLETY  [5] SPHERICITY  [6] MARGIN  [7] TEXTURE
#   [8] CALCIFICATION  [9] INTERNAL_STRUCTURE
# The model outputs z-scores; applying  actual = z * scale + mean  recovers
# the original 1-5 LIDC annotation scale.
_SCALER_MEAN = [10.9068, 2.7735, 1.5978, 1.6765, 3.7902, 3.7643, 3.8721, 4.3516, 5.6439, 1.0126]
_SCALER_SCALE = [7.1615, 0.9491, 0.9178, 0.9178, 1.0275, 0.9178, 1.0576, 1.1356, 0.9178, 0.9178]


def _inverse_scale(z: float, idx: int) -> float:
    raw = z * _SCALER_SCALE[idx] + _SCALER_MEAN[idx]
    return round(max(1.0, min(raw, 6.0)), 3)


class NoduleFeatures(BaseModel):
    malignancy: float
    spiculation: float
    lobulation: float
    subtlety: float
    sphericity: float
    margin: float
    texture: float
    calcification: float

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "NoduleFeatures":
        values = tensor.detach().cpu().numpy().flatten()
        return cls(
            malignancy=_inverse_scale(float(values[1]), 1),
            spiculation=_inverse_scale(float(values[2]), 2),
            lobulation=_inverse_scale(float(values[3]), 3),
            subtlety=_inverse_scale(float(values[4]), 4),
            sphericity=_inverse_scale(float(values[5]), 5),
            margin=_inverse_scale(float(values[6]), 6),
            texture=_inverse_scale(float(values[7]), 7),
            calcification=_inverse_scale(float(values[8]), 8),
        )
