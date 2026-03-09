from dataclasses import dataclass
import os

@dataclass
class MiddlewareConfig:
    GATEWAY_TOKEN : str = os.getenv("GATEWAY_SECRET","secret-key")
    HEALTH_PATHS: set[str] = { "/health" , "/actuator" }


@dataclass
class EurekaConfig:
    
    EUREKA_SERVER : str = os.getenv("EUREKA_SERVER",
                                     "http://eureka:eureka123@localhost:8761/eureka")
    
    APP_NAME : str = "MACHINE-LEARNING-SERVICE"

    APP_PORT : int = int(os.getenv("APP_PORT", "8000"))
    