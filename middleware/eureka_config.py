import socket
from .middleware_config import EurekaConfig
import py_eureka_client.eureka_client as eureka_client


def get_local_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


def register_eureka(config : EurekaConfig):
    eureka_client.init(
        eureka_server=config.EUREKA_SERVER,
        app_name=config.APP_NAME,
        instance_port=config.APP_PORT,
        instance_host=get_local_ip(),
    )


def deregister_eureka():
    eureka_client.stop()