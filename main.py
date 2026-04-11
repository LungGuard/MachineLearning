from fastapi import FastAPI, UploadFile, File, Form
from common.dto import PipelineResults
from common.constants import ApiConstants
from contextlib import asynccontextmanager
from middleware.gateway_auth import GatewayAuthMiddleware
from middleware.eureka_config import register_eureka, deregister_eureka
from middleware.middleware_config import EurekaConfig
from typing import Annotated


eureka_config = EurekaConfig()

@asynccontextmanager
async def lifespan(app:FastAPI):
    register_eureka(config=eureka_config)
    yield
    deregister_eureka()



app = FastAPI(
              title=ApiConstants.SERVICE_NAME,
              lifespan=lifespan
              )
app.add_middleware(GatewayAuthMiddleware)


@app.post("/api/models/analyze",response_model=PipelineResults)
def analyze(dicom_files: Annotated[
        list[UploadFile],
        File(description="CT scan DICOM series files (.dcm)"),
    ]):
    pass

