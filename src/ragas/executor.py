from __future__ import annotations

import asyncio
import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from tqdm.auto import tqdm

from ragas.exceptions import MaxRetriesExceeded
from ragas.run_config import RunConfig

logger = logging.getLogger(__name__)


def is_event_loop_running() -> bool:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    else:
        return loop.is_running()


def as_completed(coros, max_workers):
    if max_workers == -1:
        return asyncio.as_completed(coros)

    semaphore = asyncio.Semaphore(max_workers)

    async def sema_coro(coro):
        async with semaphore:
            return await coro

    sema_coros = [sema_coro(c) for c in coros]

    return asyncio.as_completed(sema_coros)


@dataclass
class Executor:
    desc: str = "Evaluating"
    keep_progress_bar: bool = True
    show_progress: bool = True
    jobs: t.List[t.Any] = field(default_factory=list, repr=False)
    raise_exceptions: bool = False
    run_config: t.Optional[RunConfig] = field(default=None, repr=False)
    _nest_asyncio_applied: bool = field(default=False, repr=False)

    def wrap_callable_with_index(self, callable: t.Callable, counter):
        async def wrapped_callable_async(*args, **kwargs):
            result = np.nan
            try:
                result = await callable(*args, **kwargs)
            except MaxRetriesExceeded as e:
                # this only for testset generation v2
                logger.warning(f"max retries exceeded for {e.evolution}")
            except Exception as e:
                if self.raise_exceptions:
                    raise e
                else:
                    exec_name = type(e).__name__
                    exec_message = str(e)
                    logger.error(
                        "Exception raised in Job[%s]: %s(%s)",
                        counter,
                        exec_name,
                        exec_message,
                        exc_info=False,
                    )

            return counter, result

        return wrapped_callable_async

    def submit(
        self, callable: t.Callable, *args, name: t.Optional[str] = None, **kwargs
    ):
        callable_with_index = self.wrap_callable_with_index(callable, len(self.jobs))
        self.jobs.append((callable_with_index, args, kwargs, name))

    def results(self) -> t.List[t.Any]:
        if is_event_loop_running():
            # an event loop is running so call nested_asyncio to fix this
            try:
                import nest_asyncio
            except ImportError:
                raise ImportError(
                    "It seems like your running this in a jupyter-like environment. Please install nest_asyncio with `pip install nest_asyncio` to make it work."
                )

            if not self._nest_asyncio_applied:
                nest_asyncio.apply()
                self._nest_asyncio_applied = True

        # create a generator for which returns tasks as they finish
        futures_as_they_finish = as_completed(
            coros=[afunc(*args, **kwargs) for afunc, args, kwargs, _ in self.jobs],
            max_workers=(self.run_config or RunConfig()).max_workers,
        )

        async def _aresults() -> t.List[t.Any]:
            results = []
            for future in tqdm(
                futures_as_they_finish,
                desc=self.desc,
                total=len(self.jobs),
                # whether you want to keep the progress bar after completion
                leave=self.keep_progress_bar,
                disable=not self.show_progress,
            ):
                r = await future
                results.append(r)

            return results

        results = asyncio.run(_aresults())
        sorted_results = sorted(results, key=lambda x: x[0])
        return [r[1] for r in sorted_results]
