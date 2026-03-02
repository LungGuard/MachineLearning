from pydantic import BaseModel

class BoundingBox(BaseModel):
    x : float
    y : float
    height : float
    width : float

