from pydantic import BaseModel
from .nodule import Nodule
from .cancer_class import CancerClass
from typing import Optional

class PipelineResults(BaseModel):
    nodules : list[Nodule]
    cancer_confidence : float
    cancer_class : Optional[CancerClass]
    notes : Optional[str]
