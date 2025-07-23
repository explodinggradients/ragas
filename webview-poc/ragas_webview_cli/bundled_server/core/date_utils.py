from enum import Enum

import pendulum


class DateFormat(str, Enum):
    SHORT = "DD-MMM-YYYY"  # eg: "01-Dec-2023"
    TIME = "hh:mm A"  # eg: "09:30 AM"
    DATE_TIME = "DD-MMM-YYYY hh:mm A"  # eg: "01-Dec-2023 09:30 AM"
    ISO_DATE_TIME = "YYYY-MM-DDTHH:mm:ss"  # eg: "2024-01-15T13:45:23"
    LOGGER_DATE_TIME = "%Y-%m-%d %H:%M:%S"  # eg: "2024-01-15 13:45:23"


def get_utc_time(*args):
    """Return current UTC time for logger"""
    return pendulum.now("UTC").timetuple()


def add_months_to_now(months: int) -> str:
    return pendulum.now().add(months=months).isoformat()


def get_current_unix_timestamp() -> int:
    return int(pendulum.now().timestamp())


def get_current_iso_datetime() -> str:
    return pendulum.now().format(DateFormat.ISO_DATE_TIME)


def format_app_date(date_string: str) -> str:
    return pendulum.parse(date_string).format(DateFormat.SHORT)


def format_app_time(date_string: str) -> str:
    return pendulum.parse(date_string).format(DateFormat.TIME)


def format_app_datetime(date_string: str) -> str:
    return pendulum.parse(date_string).format(DateFormat.DATE_TIME)


def format_iso_datetime(date_string: str) -> str:
    """Format the given date string to ISO Date-Time format (YYYY-MM-DDTHH:mm:ss)."""
    return pendulum.parse(date_string).format(DateFormat.ISO_DATE_TIME)
