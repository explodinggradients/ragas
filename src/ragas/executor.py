from __future__ import annotations

import asyncio
import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from tqdm.asyncio import tqdm as atqdm

from ragas.run_config import RunConfig
from ragas.utils import batched

logger = logging.getLogger(__name__)


def is_event_loop_running() -> bool:
    """
    Check if an event loop is currently running.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    else:
        return loop.is_running()


async def as_completed(
    coroutines: t.List[t.Coroutine], max_workers: int
) -> t.Iterator[asyncio.Future]:
    """
    Wrap coroutines with a semaphore if max_workers is specified.

    Returns an iterator of futures that completes as tasks finish.
    """
    if max_workers == -1:
        tasks = [asyncio.create_task(coro) for coro in coroutines]

    else:
        semaphore = asyncio.Semaphore(max_workers)

        async def sema_coro(coro):
            async with semaphore:
                return await coro

        tasks = [asyncio.create_task(sema_coro(coro)) for coro in coroutines]
    return asyncio.as_completed(tasks)


@dataclass
class Executor:
    """
    Executor class for running asynchronous jobs with progress tracking and error handling.

    Attributes
    ----------
    desc : str
        Description for the progress bar
    show_progress : bool
        Whether to show the progress bar
    keep_progress_bar : bool
        Whether to keep the progress bar after completion
    jobs : List[Any]
        List of jobs to execute
    raise_exceptions : bool
        Whether to raise exceptions or log them
    batch_size : int
        Whether to batch (large) lists of tasks
    run_config : RunConfig
        Configuration for the run
    _nest_asyncio_applied : bool
        Whether nest_asyncio has been applied
    """

    desc: str = "Evaluating"
    show_progress: bool = True
    keep_progress_bar: bool = True
    jobs: t.List[t.Any] = field(default_factory=list, repr=False)
    raise_exceptions: bool = False
    batch_size: t.Optional[int] = None
    run_config: t.Optional[RunConfig] = field(default=None, repr=False)
    _nest_asyncio_applied: bool = field(default=False, repr=False)

    def wrap_callable_with_index(
        self, callable: t.Callable, counter: int
    ) -> t.Callable:
        async def wrapped_callable_async(
            *args, **kwargs
        ) -> t.Tuple[int, t.Callable | float]:
            try:
                result = await callable(*args, **kwargs)
                return counter, result
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
                return counter, np.nan

        return wrapped_callable_async

    def submit(
        self,
        callable: t.Callable,
        *args,
        name: t.Optional[str] = None,
        **kwargs,
    ) -> None:
        """Submit a job to be executed, wrapping the callable with error handling and indexing to keep track of the job index."""
        callable_with_index = self.wrap_callable_with_index(callable, len(self.jobs))
        self.jobs.append((callable_with_index, args, kwargs, name))

    async def _process_batch(
        self,
        batch: t.Iterable[
            t.Tuple[t.Callable[..., t.Coroutine], t.Tuple, t.Dict, t.Optional[str]]
        ],
        max_workers: int,
        progress_bar_desc: str = "Batch progress",
    ) -> t.List[t.Tuple[int, t.Coroutine]]:
        """Process batch of jobs with optional progress tracking."""
        results = []
        coroutines = [afunc(*args, **kwargs) for afunc, args, kwargs, _ in batch]

        # Convert coroutines to futures that complete as they finish
        with atqdm(
            total=len(list(batch)),
            desc=progress_bar_desc,
            disable=not self.show_progress,
        ) as batch_pbar:
            for future in await as_completed(coroutines, max_workers):
                result = await future
                results.append(result)
                batch_pbar.update(1)
        return results

    async def _process_jobs(self) -> t.List[t.Any]:
        """Execute jobs with optional progress tracking."""
        max_workers = (self.run_config or RunConfig()).max_workers

        if self.batch_size:
            results = []
            batches = batched(self.jobs, self.batch_size)  # generator
            n_batches = (len(self.jobs) + self.batch_size - 1) // self.batch_size

            with atqdm(  # outer progress bar
                total=n_batches,
                desc=self.desc,
                disable=not self.show_progress,
            ) as pbar:
                for i, batch in enumerate(batches, 1):
                    batch_results = await self._process_batch(
                        batch,
                        max_workers=max_workers,
                        progress_bar_desc=f"Batch {i}/{n_batches}",
                    )
                    results.extend(batch_results)
                    pbar.update(1)

        else:
            results = await self._process_batch(
                self.jobs,
                max_workers=max_workers,
                progress_bar_desc=self.desc,
            )

        return results

    def results(self) -> t.List[t.Any]:
        """
        Execute all submitted jobs and return their results.

        The results are returned in the order of job submission.
        """
        if is_event_loop_running():
            # an event loop is running so call nested_asyncio to fix this
            try:
                import nest_asyncio
            except ImportError as e:
                raise ImportError(
                    "It seems like your running this in a jupyter-like environment. "
                    "Please install nest_asyncio with `pip install nest_asyncio` to make it work."
                ) from e

            if not self._nest_asyncio_applied:
                nest_asyncio.apply()
                self._nest_asyncio_applied = True

        results = asyncio.run(self._process_jobs())
        sorted_results = sorted(results, key=lambda x: x[0])
        return [r[1] for r in sorted_results]


def run_async_batch(
    desc: str,
    func: t.Callable,
    kwargs_list: t.List[t.Dict],
    batch_size: t.Optional[int] = None,
):
    """Provide functionality to run the same async function with different arguments in parallel."""
    run_config = RunConfig()
    executor = Executor(
        desc=desc,
        keep_progress_bar=False,
        raise_exceptions=True,
        run_config=run_config,
        batch_size=batch_size,
    )

    for kwargs in kwargs_list:
        executor.submit(func, **kwargs)

    return executor.results()
