import shutil
import tempfile
from typing import Callable, Protocol, runtime_checkable

from common.constants.actuator import ActuatorStatus
from common.dto.actuator import HealthComponent

_DISK_THRESHOLD_BYTES = 10 * 1024 * 1024  # 10 MiB — Spring Boot default


@runtime_checkable
class HealthIndicator(Protocol):
    def check(self) -> HealthComponent: ...


class PingHealthIndicator:
    def check(self) -> HealthComponent:
        return HealthComponent(status=ActuatorStatus.UP)


class DiskSpaceHealthIndicator:
    def __init__(self, path: str = None, threshold_bytes: int = _DISK_THRESHOLD_BYTES):
        self._path = path or tempfile.gettempdir()
        self._threshold = threshold_bytes

    def check(self) -> HealthComponent:
        usage = shutil.disk_usage(self._path)
        status = ActuatorStatus.UP if usage.free > self._threshold else ActuatorStatus.DOWN
        return HealthComponent(
            status=status,
            details={
                "total": usage.total,
                "free": usage.free,
                "threshold": self._threshold,
                "path": self._path,
            },
        )


class ModelsHealthIndicator:
    def __init__(self, pipeline_factory: Callable):
        self._pipeline_factory = pipeline_factory

    def check(self) -> HealthComponent:
        try:
            self._pipeline_factory()
            return HealthComponent(
                status=ActuatorStatus.UP,
                details={"detail": "pipeline ready"},
            )
        except NotImplementedError:
            return HealthComponent(
                status=ActuatorStatus.OUT_OF_SERVICE,
                details={"detail": "pipeline not yet wired"},
            )
        except Exception as exc:
            return HealthComponent(
                status=ActuatorStatus.DOWN,
                details={"error": str(exc)[:200]},
            )
