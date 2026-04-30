import datetime

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from common.constants import Header, StatusCode

from .middleware_config import MiddlewareConfig
from .middleware_contants import GatewayAuthConstants

config = MiddlewareConfig()


class GatewayAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if self._is_health_path(request) or self._is_authorized(request):
            return await call_next(request)

        return JSONResponse(
            status_code=StatusCode.FORBIDDEN,
            content={
                GatewayAuthConstants.ERROR_RESPONSE_FIELD : GatewayAuthConstants.ERROR_RESPONSE,
                GatewayAuthConstants.MESSAGE_RESPONSE_FIELD : GatewayAuthConstants.MESSAGE_RESPONSE ,
                GatewayAuthConstants.TIMESTAMP_RESPONSE_FIELD : datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
        )

    @staticmethod
    def _is_health_path(request: Request) -> bool:
        return any(request.url.path.startswith(path) for path in config.HEALTH_PATHS)

    @staticmethod
    def _is_authorized(request: Request) -> bool:
        return request.headers.get(Header.GATEWAY) == config.GATEWAY_TOKEN
