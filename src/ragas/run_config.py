import typing as t
from dataclasses import dataclass

from tenacity import (
    AsyncRetrying,
    Retrying,
    WrappedFn,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log
)

import logging

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(threadName)s] - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

@dataclass
class RunConfig:
    """
    Configuration for a timeouts and retries.
    """

    timeout: int = 25
    max_retries: int = 5
    max_wait: int = 5
    max_workers: int = 4
    exception_types: t.Union[
        t.Type[BaseException],
        t.Tuple[t.Type[BaseException], ...],
    ] = Exception


def add_retry(fn: WrappedFn, run_config: RunConfig) -> WrappedFn:
    r = Retrying(
        wait=wait_random_exponential(multiplier=1, max=run_config.max_wait),
        stop=stop_after_attempt(run_config.max_retries),
        retry=retry_if_exception_type(run_config.exception_types),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.DEBUG),
    )
    return r.wraps(fn)


def add_async_retry(fn: WrappedFn, run_config: RunConfig) -> WrappedFn:
    """
    Decorator for retrying a function if it fails.
    """
    r = AsyncRetrying(
        wait=wait_random_exponential(multiplier=1, max=run_config.max_wait),
        stop=stop_after_attempt(run_config.max_retries),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.DEBUG)
    )
    return r.wraps(fn)
