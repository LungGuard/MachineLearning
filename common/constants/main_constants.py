class ScanConstants:
    ID_MIN_LENGTH = 1
    ID_MAX_LENGTH = 128
    ID_PATTERN = r"^[A-Za-z0-9_-]+$"
    DIR_PREFIX_TEMPLATE = "scan_{scan_id}_"
    ZIP_NAME = "scan.zip"
    EXTRACT_DIR_NAME = "extracted"
    DICOM_GLOB = "*.dcm"


class ErrorSlugs:
    VALIDATION = "validation_error"
    INTERNAL = "internal_error"
    DEFAULT = "error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    


class PipelineMessages:
    UNAVAILABLE = "pipeline not yet available"
    FAILURE_TEMPLATE = "pipeline failure for scan {scan_id}"
