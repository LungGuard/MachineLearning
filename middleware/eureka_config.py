import logging
import socket
import uuid

import py_eureka_client.eureka_client as eureka_client

from common.constants.api_constants import ApiConstants

from .middleware_config import EurekaConfig

logger = logging.getLogger(__name__)


def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            logger.info("Resolved local IP via UDP probe: %s", ip)
            return ip
    except OSError:
        fallback = socket.gethostname()
        logger.info("UDP IP probe failed; falling back to hostname: %s", fallback)
        return fallback


async def register_eureka(config: EurekaConfig):
    host = get_local_ip()
    base_url = f"http://{host}:{config.APP_PORT}"
    instance_id = f"{config.APP_NAME}:{host}:{config.APP_PORT}:{uuid.uuid4().hex[:8]}"

    await eureka_client.init_async(
        eureka_server=config.EUREKA_SERVER,
        app_name=config.APP_NAME,
        instance_port=config.APP_PORT,
        instance_host=host,
        instance_id=instance_id,
        home_page_url=base_url,
        status_page_url=f"{base_url}/actuator/info",
        health_check_url=f"{base_url}/actuator/health",
        renewal_interval_in_secs=config.RENEWAL_INTERVAL_IN_SECS,
        duration_in_secs=config.LEASE_DURATION_IN_SECS,
        metadata={
            "version": ApiConstants.VERSION,
            "framework": "fastapi",
        },
    )


async def deregister_eureka():
    await eureka_client.stop_async()
