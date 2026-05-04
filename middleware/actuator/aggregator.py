from common.constants.actuator import ActuatorStatus
from common.dto.actuator import HealthComponent


_PRECEDENCE = [
    ActuatorStatus.DOWN,
    ActuatorStatus.OUT_OF_SERVICE,
    ActuatorStatus.UNKNOWN,
    ActuatorStatus.UP,
]


def aggregate_status(components: dict[str, HealthComponent]) -> ActuatorStatus:
    if not components:
        return ActuatorStatus.UNKNOWN
    statuses = {c.status for c in components.values()}
    for status in _PRECEDENCE:
        if status in statuses:
            return status
    return ActuatorStatus.UNKNOWN
