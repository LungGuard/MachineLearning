from .middleware_config import MiddlewareConfig
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from common.constants import Header,StatusCode
import datetime

config = MiddlewareConfig()

class GatewayAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request : Request, call_next):
        
        is_health = any(request.url.path.startswith(path) for path in config.HEALTH_PATHS)
        is_authorized = request.headers.get(Header.GATEWAY) == config.GATEWAY_TOKEN
        
        return (
            await call_next(request) 
            if (is_health and is_authorized) 
            else 
            JSONResponse(
                status_code=StatusCode.FORBBIDEN,
                content={
                    "error": "forbbiden",
                    "message" : "Direct access not allowed, use API Gateway",
                    "timestamp" : str(datetime.datetime())
                }
            )
        )