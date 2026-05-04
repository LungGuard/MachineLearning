from fastapi import Request

from common.constants import Header
from common.dto import error_response

from .middleware_config import MiddlewareConfig

config = MiddlewareConfig()


class GatewayAuthMiddleware:
    """Pure ASGI middleware — avoids BaseHTTPMiddleware's exception-swallowing."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        if self._is_health_path(request) or self._is_authorized(request):
            await self.app(scope, receive, send)
            return

        response = error_response(403, "forbidden", "Direct access not allowed, use API Gateway")
        await response(scope, receive, send)

    @staticmethod
    def _is_health_path(request: Request) -> bool:
        return any(request.url.path.startswith(path) for path in config.HEALTH_PATHS)

    @staticmethod
    def _is_authorized(request: Request) -> bool:
        return request.headers.get(Header.GATEWAY) == config.GATEWAY_TOKEN
