from pydantic import BaseModel
from.bbox import BoundingBox
from DetectionModel.constants.dataclasses import NoduleFeatures

class Nodule(BaseModel):
    nodule_id : str
    bbox : BoundingBox
    confidence : float
    nodule_features : NoduleFeatures
    nodule_image : str #base64 encoded image
    
