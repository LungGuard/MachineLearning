from pydantic import BaseModel
from ClassificationModel.constants.enums import CancerType
from typing import Union

class CancerClass(BaseModel):
    cancer_type: Union[CancerType,str]
    confidnece : float 