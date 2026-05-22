import torch
from pydantic import BaseModel


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
        # Features.getNoduleFeaturesVector() order:
        # [0] DIAMETER_MM  [1] MALIGNANCY  [2] SPICULATION  [3] LOBULATION
        # [4] SUBTLETY  [5] SPHERICITY  [6] MARGIN  [7] TEXTURE
        # [8] CALCIFICATION  [9] INTERNAL_STRUCTURE
        return cls(
            malignancy=float(values[1]),
            spiculation=float(values[2]),
            lobulation=float(values[3]),
            subtlety=float(values[4]),
            sphericity=float(values[5]),
            margin=float(values[6]),
            texture=float(values[7]),
            calcification=float(values[8]),
        )
