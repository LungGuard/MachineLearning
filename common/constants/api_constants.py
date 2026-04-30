from dataclasses import dataclass

class ApiConstants:
    SERVICE_NAME = "MachineLearningService"
    MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024 #2gb

@dataclass
class DownloadLimits:
    connect: float  = 10.0
    read : float = 120.0
    write : float = 30.0
    pool : float = 10.0
    