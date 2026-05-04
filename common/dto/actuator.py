from typing import Any

from pydantic import BaseModel

from common.constants.actuator import ActuatorStatus


class HealthComponent(BaseModel):
    status: ActuatorStatus
    details: dict[str, Any] = {}


class HealthResponse(BaseModel):
    status: ActuatorStatus
    components: dict[str, HealthComponent]


class InfoResponse(BaseModel):
    app: dict[str, str]
    build: dict[str, str]
    runtime: dict[str, str]
