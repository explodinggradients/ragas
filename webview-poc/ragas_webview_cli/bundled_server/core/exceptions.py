from typing import Optional

from ..boot import app_config, app_log


def exception(
    debug_message: str,
    api_message: Optional[str] = "An internal server error occured",
    exc: Optional[Exception] = None,
) -> Exception:
    """
    Returns and logs exceptions based on environment settings.

    Args:
        debug_message: Internal debugging message (not shown to end users)
        api_message: User-facing error message
        exc: Original exception to rethrow if in development

    Returns:
        Exception: Original exception in development, or new exception with api_message in production
    """
    is_development = app_config.is_development()

    log_message = (
        f"[Exception] [DEVELOPMENT MODE] An exception occurred \n"
        f"API Message: {api_message} \n"
        f"Debug Message: {debug_message} \n"
    )

    if is_development:
        # In development, if we have original exception, return it with full logging
        if exc is not None and isinstance(exc, Exception):
            app_log.error(
                f"{log_message}, Exception: {str(exc)}",
                exc_info=True,
            )
            return exc

        # In development without exception, return new exception with debug details
        app_log.error(
            log_message,
            exc_info=True,
        )
        return Exception(log_message)

    # In production, log everything but return only user-facing message
    app_log.error(
        log_message,
    )

    if exc is not None and isinstance(exc, Exception):
        app_log.debug(
            f"An Exception was captured in `exceptions.exception()`: {str(exc)}",
            exc_info=True,
        )
    return Exception(api_message)
