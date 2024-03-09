from __future__ import annotations
import sys

import asyncio
import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from tqdm.auto import tqdm

from ragas.exceptions import MaxRetriesExceeded
from ragas.run_config import RunConfig

logger = logging.getLogger(__name__)


def as_completed(
        loop: asyncio.AbstractEventLoop,
        coros: t.Iterable[asyncio.Future],
        max_workers: int
    ):
    # support Python < 3.10 where loop argument is still required
    loop_arg_dict = {"loop": loop} if sys.version_info[:2] < (3, 10) else {}
    if max_workers == -1:
        return asyncio.as_completed(coros, **loop_arg_dict)  # type: ignore
    
    semaphore = asyncio.Semaphore(max_workers, **loop_arg_dict)
    async def sema_coro(coro: asyncio.Future):
        async with semaphore:
            return await coro
    
    sema_coros = [sema_coro(c) for c in coros]
    return asyncio.as_completed(sema_coros, **loop_arg_dict)  # type: ignore

@dataclass
class Executor:
    desc: str = "Evaluating"
    keep_progress_bar: bool = True
    jobs: t.List[t.Any] = field(default_factory=list, repr=False)
    raise_exceptions: bool = False
    run_config: t.Optional[RunConfig] = None

    def wrap_callable_with_index(self, callable: t.Callable, counter):
        async def wrapped_callable_async(*args, **kwargs):
            return counter, await callable(*args, **kwargs)

        return wrapped_callable_async

    def submit(
        self, callable: t.Callable, *args, name: t.Optional[str] = None, **kwargs
    ):
        callable_with_index = self.wrap_callable_with_index(callable, len(self.jobs))
        self.jobs.append((callable_with_index, args, kwargs, name))

    def results(self) -> t.List[t.Any]:
        loop = asyncio.get_event_loop()
        
        futures = as_completed(
            loop=loop,
            coros=[afunc(*args, **kwargs) for afunc, args, kwargs, _ in self.jobs],
            max_workers=(self.run_config or RunConfig()).max_workers
        )
        results = loop.run_until_complete(self._aresults(futures))

        if results is None:
            if self.raise_exceptions:
                raise RuntimeError(
                    "Executor failed to complete. Please check logs above for full info."
                )
            else:
                logger.error("Executor failed to complete. Please check logs above.")
                return []
        sorted_results = sorted(results, key=lambda x: x[0])
        return [r[1] for r in sorted_results]

    async def _aresults(
            self,
            futures: t.Iterator[asyncio.Future]
        ) -> t.List[t.Any]:
        results = []
        for future in tqdm(
            futures,
            desc=self.desc,
            total=len(self.jobs),
            # whether you want to keep the progress bar after completion
            leave=self.keep_progress_bar,
        ):
            r = (-1, np.nan)
            try:
                r = await future
            except MaxRetriesExceeded as e:
                logger.warning(f"max retries exceeded for {e.evolution}")
            except Exception as e:
                if self.raise_exceptions:
                    raise e
                else:
                    logger.error(
                        "Runner in Executor raised an exception", exc_info=True
                    )
            results.append(r)

        return results