import os
from dataclasses import dataclass, field


def _require_env(name: str) -> str:
    if value := os.getenv(name):
        return value
    else:
        raise RuntimeError(
            f"Required environment variable {name!r} is not set. "
            "Refusing to start with insecure defaults."
        )


@dataclass
class MiddlewareConfig:
    GATEWAY_TOKEN: str = field(default_factory=lambda: _require_env("GATEWAY_SECRET"))
    HEALTH_PATHS: frozenset[str] = field(
        default_factory=lambda: frozenset({"/health", "/actuator"})
    )


@dataclass
class EurekaConfig:
    EUREKA_SERVER: str = field(default_factory=lambda: _require_env("EUREKA_SERVER"))
    APP_NAME: str = "MACHINE-LEARNING-SERVICE"
    APP_PORT: int = field(default_factory=lambda: int(os.getenv("APP_PORT", "8000")))
