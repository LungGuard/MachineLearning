import datetime

from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: str


def error_response(status_code: int, error: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error=error,
            message=message,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        ).model_dump(),
    )
