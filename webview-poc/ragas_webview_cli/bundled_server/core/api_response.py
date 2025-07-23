from typing import Any, Generic, Optional, TypeVar

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..boot import app_config, app_log

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """
    Standard API response model that provides consistent response structure.

    Attributes:
        status (str): Response status - "success" or "error"
        status_code (int): HTTP status code
        message (str): Human readable message
        data (Optional[T]): Response payload data, can be None
        debug_error_info (Optional[Any]): Debug information, only in development mode
    """

    status: str
    status_code: int
    message: str
    data: Optional[T] = None
    debug_error_info: Optional[Any] = None

    @staticmethod
    def success(
        data: Optional[Any] = None,
        message: Optional[str] = None,
        status_code: int = 200,
    ) -> JSONResponse:
        """
        Creates a success response.

        Args:
            data: Response payload
            message: Optional success message
            status_code: HTTP status code, defaults to 200

        Returns:
            JSONResponse: FastAPI response object with standardized structure
        """
        return JSONResponse(
            status_code=status_code,
            content={
                "status": "success",
                "status_code": status_code,
                "message": message,
                "data": data,
            },
        )

    @staticmethod
    def exception(
        status_code: int = 400,
        api_message: Optional[str] = "An internal server error occured",
        debug_message: Optional[str] = None,
        exc: Optional[Exception] = None,
        rethrow_if_exc_http_exception: Optional[bool] = False,
    ) -> HTTPException:
        """
        Creates a standardized HTTP exception with optional debug information in development environment.

        Args:
            api_message: User-facing error message that will be shown in all environments
            status_code: HTTP status code for the response (defaults to 400 Bad Request)
            debug_message: Internal debugging message only shown in development environment
            exc: Original exception to include in error details in development environment
            rethrow_if_exc_http_exception: If True and exc is HTTPException, returns original exception unchanged

        Returns:
            HTTPException with appropriate error details based on environment:
            - Production: Only contains api_message
            - Development: Includes debug_message and exception details

        Example Usage:
            raise ApiResponse.exception(
                api_message="Invalid input provided",
                status_code=400,
                debug_message="[validate_input] User provided malformed JSON",
                exc=original_exception
                rethrow_if_exc_http_exception=False
            )
        """

        # If configured to rethrow and exception is already HTTPException, return it unchanged
        if (
            rethrow_if_exc_http_exception
            and exc is not None
            and isinstance(exc, HTTPException)
        ):
            return exc

        is_development = app_config.is_development()

        log_message = (
            f"[HTTPException] [DEVELOPMENT MODE] An ApiResponse.exception occurred \n"
            f"API Message: {api_message} \n"
            f"Debug Message: {debug_message} \n"
        )

        if is_development:
            # In development, if we have original exception, return a HTTPException with full logging
            if exc is not None and isinstance(exc, Exception):
                msg = f"{log_message}, Exception: {str(exc)}"
                app_log.error(
                    msg,
                    exc_info=True,
                )
                return HTTPException(
                    status_code=status_code,
                    detail=msg,
                )

            # In development without exception, return new HTTPException with debug details
            app_log.error(
                log_message,
                exc_info=True,
            )
            return HTTPException(
                status_code=status_code,
                detail=log_message,
            )

        # In production, log everything but return only user-facing message
        app_log.error(
            log_message,
        )

        if exc is not None and isinstance(exc, Exception):
            app_log.error(
                f"An Exception was captured in `ApiResponse.exception()`: {str(exc)}",
                exc_info=True,
            )

        return HTTPException(
            status_code=status_code,
            detail=api_message,
        )

    @classmethod
    def internal_error_transformer(
        cls,
        status_code: int,
        message: str,
        error: Optional[Exception] = None,
    ) -> "ApiResponse":
        """
        Internal method to transform errors into ApiResponse format.
        Do not use directly for returning errors, use `ApiResponse.exception()` instead.

        Args:
            status_code: HTTP error status code
            message: Error message to show to user
            error: Optional exception object for debug information

        Returns:
            ApiResponse: Error response with optional debug information in development

        Note:
            This is a transformer method only, not meant for direct error responses.
            Always use HTTPException for raising errors which will be transformed
            by the global exception handler.
        """
        response = cls(
            status="error", status_code=status_code, message=message, data=None
        )

        # Add debug info only in development mode
        if app_config.is_development() and error:
            response.debug_error_info = {
                "error_type": error.__class__.__name__,
                "error_message": str(error),
                "error_details": getattr(error, "detail", None),
            }

        return response
