from __future__ import annotations

import logging
import os
import typing as t
import warnings
from functools import lru_cache

import numpy as np

if t.TYPE_CHECKING:
    from ragas.metrics.base import Metric
    from ragas.testset.evolutions import Evolution

DEBUG_ENV_VAR = "RAGAS_DEBUG"


@lru_cache(maxsize=1)
def get_cache_dir() -> str:
    "get cache location"
    DEFAULT_XDG_CACHE_HOME = "~/.cache"
    xdg_cache = os.getenv("XDG_CACHE_HOME", DEFAULT_XDG_CACHE_HOME)
    default_ragas_cache = os.path.join(xdg_cache, "ragas")
    return os.path.expanduser(os.getenv("RAGAS_CACHE_HOME", default_ragas_cache))


@lru_cache(maxsize=1)
def get_debug_mode() -> bool:
    if os.environ.get(DEBUG_ENV_VAR, str(False)).lower() == "true":
        return True
    else:
        return False


def check_if_sum_is_close(
    values: t.List[float], close_to: float, num_places: int
) -> bool:
    multiplier = 10**num_places
    total = sum(int(round(v * multiplier)) for v in values)
    return total == int(round(close_to * multiplier))


def patch_logger(module: str, level: int):
    # enable debug logging
    patched_logger = logging.getLogger(module)
    patched_logger.setLevel(level=level)
    # Create a handler for the asyncio logger
    handler = logging.StreamHandler()  # or another type of Handler
    handler.setLevel(logging.DEBUG)
    # Optional: Set a formatter if you want a specific format for the logs
    formatter = logging.Formatter("[%(name)s.%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    # Add the handler to the asyncio logger
    patched_logger.addHandler(handler)
    # Set propagate to False if you don't want it to log to the root logger's handlers as well
    patched_logger.propagate = False


# Function to check if an element is NaN
def is_nan(x):
    try:
        return np.isnan(x)
    except TypeError:
        return False


def get_feature_language(feature: t.Union[Metric, Evolution]) -> t.Optional[str]:
    from ragas.llms.prompt import Prompt

    languags = [
        value.language
        for _, value in vars(feature).items()
        if isinstance(value, Prompt)
    ]
    return languags[0] if len(languags) > 0 else None


def deprecated(
    since: str,
    *,
    removal: t.Optional[str] = None,
    alternative: t.Optional[str] = None,
    addendum: t.Optional[str] = None,
    pending: bool = False,
):
    """
    Decorator to mark functions or classes as deprecated.

    Args:
        since: str
             The release at which this API became deprecated.
        removal: str, optional
            The expected removal version. Cannot be used with pending=True.
            Must be specified with pending=False.
        alternative: str, optional
            The alternative API or function to be used instead
            of the deprecated function.
        addendum: str, optional
            Additional text appended directly to the final message.
        pending: bool
            Whether the deprecation version is already scheduled or not.
            Cannot be used with removal.


    Examples
    --------

        .. code-block:: python

            @deprecated("0.1", removal="0.2", alternative="some_new_function")
            def some_old_function():
                print("This is an old function.")

    """

    def deprecate(func: t.Callable):
        def emit_warning(*args, **kwargs):
            if pending and removal:
                raise ValueError(
                    "A pending deprecation cannot have a scheduled removal"
                )

            message = f"The function {func.__name__} was deprecated in {since},"

            if not pending:
                if removal:
                    message += f" and will be removed in the {removal} release."
                else:
                    raise ValueError(
                        "A non-pending deprecation must have a scheduled removal."
                    )
            else:
                message += " and will be removed in a future release."

            if alternative:
                message += f" Use {alternative} instead."

            if addendum:
                message += f" {addendum}"

            warnings.warn(message, stacklevel=2, category=DeprecationWarning)
            return func(*args, **kwargs)

        return emit_warning

    return deprecate


def get_or_init(
    dictionary: t.Dict[str, t.Any], key: str, default: t.Callable[[], t.Any]
) -> t.Any:
    _value = dictionary.get(key)
    value = _value if _value is not None else default()

    return value
