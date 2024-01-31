import typing as t
from dataclasses import dataclass

from tenacity import (
    AsyncRetrying,
    Retrying,
    wait_exponential,
    stop_after_attempt,
)


@dataclass
class RunConfig:
    """
    Configuration for a timeouts and retries.
    """

    timeout: int = 60
    max_retries: int = 10
    max_wait: int = 60


def make_retry_wrapper(run_config: RunConfig) -> t.Callable:
    return Retrying(
        wait=wait_exponential(multiplier=1, max=run_config.max_wait),
        stop=stop_after_attempt(run_config.max_retries),
        reraise=True,
    )


def make_async_retry_wrapper(run_config: RunConfig) -> t.Callable:
    """
    Decorator for retrying a function if it fails.
    """
    return AsyncRetrying(
        wait=wait_exponential(multiplier=1, max=run_config.max_wait),
        stop=stop_after_attempt(run_config.max_retries),
        reraise=True,
    )
