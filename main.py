from fastapi import FastAPI
from common.dto import PipelineResults
from common.constants import ApiConstants
from py_eureka_client import eureka_client

app = FastAPI(title=ApiConstants.SERVICE_NAME) 

app_eureka_client = eureka_client(
    
)

@app.post("/api/analyze",response_model=PipelineResults)
def analyze():
    pass

