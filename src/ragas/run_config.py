from dataclasses import dataclass

from tenacity import (
    AsyncRetrying,
    Retrying,
    WrappedFn,
    stop_after_attempt,
    wait_random_exponential,
)


@dataclass
class RunConfig:
    """
    Configuration for a timeouts and retries.
    """

    timeout: int = 60
    max_retries: int = 10
    max_wait: int = 60


def add_retry(run_config: RunConfig, fn: WrappedFn) -> WrappedFn:
    r = Retrying(
        wait=wait_random_exponential(multiplier=1, max=run_config.max_wait),
        stop=stop_after_attempt(run_config.max_retries),
        reraise=True,
    )
    return r.wraps(fn)


def add_async_retry(run_config: RunConfig, fn: WrappedFn) -> WrappedFn:
    """
    Decorator for retrying a function if it fails.
    """
    r = AsyncRetrying(
        wait=wait_random_exponential(multiplier=1, max=run_config.max_wait),
        stop=stop_after_attempt(run_config.max_retries),
        reraise=True,
    )
    return r.wraps(fn)
