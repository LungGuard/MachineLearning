import platform
import sys
from typing import Callable

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from common.constants.actuator import ActuatorStatus
from common.constants.api_constants import ApiConstants
from common.dto.actuator import HealthComponent, HealthResponse, InfoResponse

from .aggregator import aggregate_status
from .health_indicators import DiskSpaceHealthIndicator, ModelsHealthIndicator, PingHealthIndicator


def build_actuator_router(pipeline_factory: Callable) -> APIRouter:
    router = APIRouter()

    ping = PingHealthIndicator()
    disk = DiskSpaceHealthIndicator()
    models = ModelsHealthIndicator(pipeline_factory)

    _indicators = {
        "ping": ping,
        "diskSpace": disk,
        "models": models,
    }

    def _build_health() -> HealthResponse:
        components = {name: ind.check() for name, ind in _indicators.items()}
        return HealthResponse(
            status=aggregate_status(components),
            components=components,
        )

    @router.get("/health", response_model=HealthResponse, tags=["actuator"])
    def health_alias():
        return _build_health()

    @router.get("/actuator", tags=["actuator"])
    def actuator_index():
        return JSONResponse(content={
            "_links": {
                "self":    {"href": "/actuator", "templated": False},
                "health":  {"href": "/actuator/health", "templated": False},
                "health-component": {
                    "href": "/actuator/health/{component}",
                    "templated": True,
                },
                "info":    {"href": "/actuator/info", "templated": False},
            }
        })

    @router.get("/actuator/health", response_model=HealthResponse, tags=["actuator"])
    def actuator_health():
        return _build_health()

    @router.get("/actuator/health/{component}", response_model=HealthComponent, tags=["actuator"])
    def actuator_health_component(component: str):
        if component not in _indicators:
            raise HTTPException(
                status_code=404,
                detail=f"health component '{component}' not found",
            )
        return _indicators[component].check()

    @router.get("/actuator/info", response_model=InfoResponse, tags=["actuator"])
    def actuator_info():
        return InfoResponse(
            app={
                "name": ApiConstants.SERVICE_NAME,
                "version": ApiConstants.VERSION,
            },
            build={
                "commit": "unknown",
                "time": "unknown",
            },
            runtime={
                "python_version": sys.version,
                "platform": platform.platform(),
                "implementation": platform.python_implementation(),
            },
        )

    return router
