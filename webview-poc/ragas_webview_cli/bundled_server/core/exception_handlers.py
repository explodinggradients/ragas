from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from ..boot import app_log
from ..core.common import DEFAULT_ERROR_STATUS
from .api_response import ApiResponse


def global_exception_handler(
    _request: Request,
    exc: Exception,
) -> JSONResponse:
    if exc is not None and isinstance(exc, HTTPException):
        error_response = ApiResponse.internal_error_transformer(
            status_code=exc.status_code,
            message=str(exc.detail),
            error=exc,
        )
        status_code = exc.status_code

        app_log.error(f"[HTTPException] error: {str(exc)}", exc_info=True)
    else:
        error_response = ApiResponse.internal_error_transformer(
            status_code=DEFAULT_ERROR_STATUS,
            message="An internal server error occured",
            error=exc,
        )
        status_code = DEFAULT_ERROR_STATUS

        app_log.error(
            f"[Exception] error: {str(exc)}, status_code: `{status_code}`, api message: `{error_response}`",
            exc_info=True,
        )

    return JSONResponse(
        status_code=status_code,
        content=error_response.model_dump(exclude_none=True),
    )
