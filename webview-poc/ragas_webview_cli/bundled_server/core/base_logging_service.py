import logging
from enum import Enum

from ..core.date_utils import DateFormat, get_utc_time

class LogLevel(Enum):
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"
    NOTSET = "NOTSET"

    @classmethod
    def to_list(cls) -> list[str]:
        return [e.value for e in cls]


def setup_base_logger(
    name: str,
    log_level: LogLevel,
) -> logging.Logger:
    """Setup logger with consistent formatting to stdout"""
    _log_level = log_level.value
    log_format = "[%(asctime)s UTC] [%(levelname)s] [%(name)s] %(message)s"

    # Create formatter to be reused
    formatter = logging.Formatter(log_format, datefmt=DateFormat.LOGGER_DATE_TIME)
    formatter.converter = get_utc_time

    # Set up basic logging
    logging.Formatter.converter = get_utc_time
    logging.basicConfig(
        format=log_format,
        level=_log_level,
        datefmt=DateFormat.LOGGER_DATE_TIME,
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(_log_level)


    return logging.getLogger(name)
