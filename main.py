from fastapi import FastAPI
from common.dto import PipelineResults
from common.constants import ApiConstants
from contextlib import asynccontextmanager
from middleware.gateway_auth import GatewayAuthMiddleware
from middleware.eureka_config import register_eureka, deregister_eureka
from middleware.middleware_config import EurekaConfig

config = EurekaConfig()

@asynccontextmanager
async def lifespan(app:FastAPI):
    register_eureka(config=config)
    yield
    deregister_eureka()



app = FastAPI(
              title=ApiConstants.SERVICE_NAME,
              lifespan=lifespan
              )
app.add_middleware(GatewayAuthMiddleware)



@app.post("/api/analyze",response_model=PipelineResults)
def analyze():
    pass

