from pydantic import BaseModel
import torch


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
    def from_tensor(cls, tensor: torch.Tensor):
        values = tensor.detach().cpu().numpy().flatten()
        return cls(
            malignancy=float(values[0]),
            spiculation=float(values[1]),
            lobulation=float(values[2]),
            subtlety=float(values[3]),
            sphericity=float(values[4]),
            margin=float(values[5]),
            texture=float(values[6]),
            calcification=float(values[7])
        )
