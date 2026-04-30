from enum import IntEnum


class StatusCode(IntEnum):
    OK = 200
    FORBIDDEN = 403
    INTERNAL_SERVER_ERROR = 500
    UNPROCESSABLE_CONTENT = 422
    BAD_GATEWAY = 502
