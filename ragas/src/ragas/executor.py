from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from tqdm.auto import tqdm

from ragas.async_utils import apply_nest_asyncio, as_completed, process_futures, run
from ragas.run_config import RunConfig
from ragas.utils import ProgressBarManager, batched

logger = logging.getLogger(__name__)


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
    pbar: t.Optional[tqdm] = None
    _jobs_processed: int = field(default=0, repr=False)

    def wrap_callable_with_index(
        self, callable: t.Callable, counter: int
    ) -> t.Callable:
        async def wrapped_callable_async(*args, **kwargs) -> t.Tuple[int, t.Any]:
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
        """
        Submit a job to be executed, wrapping the callable with error handling and indexing to keep track of the job index.
        """
        # Use _jobs_processed for consistent indexing across multiple runs
        callable_with_index = self.wrap_callable_with_index(
            callable, self._jobs_processed
        )
        self.jobs.append((callable_with_index, args, kwargs, name))
        self._jobs_processed += 1

    def clear_jobs(self) -> None:
        """Clear all submitted jobs and reset counter."""
        self.jobs.clear()
        self._jobs_processed = 0

    async def _process_jobs(self) -> t.List[t.Any]:
        """Execute jobs with optional progress tracking."""
        if not self.jobs:
            return []

        # Make a copy of jobs to process and clear the original list to prevent re-execution
        jobs_to_process = self.jobs.copy()
        self.jobs.clear()

        max_workers = (
            self.run_config.max_workers
            if self.run_config and hasattr(self.run_config, "max_workers")
            else -1
        )
        results = []
        pbm = ProgressBarManager(self.desc, self.show_progress)

        if not self.batch_size:
            # Use external progress bar if provided, otherwise create one
            if self.pbar is None:
                with pbm.create_single_bar(len(jobs_to_process)) as internal_pbar:
                    await self._process_coroutines(
                        jobs_to_process, internal_pbar, results, max_workers
                    )
            else:
                await self._process_coroutines(
                    jobs_to_process, self.pbar, results, max_workers
                )
            return results

        # Process jobs in batches with nested progress bars
        await self._process_batched_jobs(jobs_to_process, pbm, max_workers, results)
        return results

    async def _process_batched_jobs(
        self, jobs_to_process, progress_manager, max_workers, results
    ):
        """Process jobs in batches with nested progress tracking."""
        batch_size = self.batch_size or len(jobs_to_process)
        batches = batched(jobs_to_process, batch_size)
        overall_pbar, batch_pbar, n_batches = progress_manager.create_nested_bars(
            len(jobs_to_process), batch_size
        )

        with overall_pbar, batch_pbar:
            for i, batch in enumerate(batches, 1):
                progress_manager.update_batch_bar(batch_pbar, i, n_batches, len(batch))

                # Create coroutines per batch
                coroutines = [
                    afunc(*args, **kwargs) for afunc, args, kwargs, _ in batch
                ]
                async for result in process_futures(
                    as_completed(coroutines, max_workers), batch_pbar
                ):
                    # Ensure result is always a tuple (counter, value)
                    if isinstance(result, Exception):
                        # Find the counter for this failed job
                        idx = coroutines.index(result.__context__)
                        counter = (
                            batch[idx][0].__closure__[1].cell_contents
                        )  # counter from closure
                        results.append((counter, result))
                    else:
                        results.append(result)
                # Update overall progress bar for all futures in this batch
                overall_pbar.update(len(batch))

    async def _process_coroutines(self, jobs, pbar, results, max_workers):
        """Helper function to process coroutines and update the progress bar."""
        coroutines = [afunc(*args, **kwargs) for afunc, args, kwargs, _ in jobs]
        async for result in process_futures(
            as_completed(coroutines, max_workers), pbar
        ):
            # Ensure result is always a tuple (counter, value)
            if isinstance(result, Exception):
                idx = coroutines.index(result.__context__)
                counter = (
                    jobs[idx][0].__closure__[1].cell_contents
                )  # counter from closure
                results.append((counter, result))
            else:
                results.append(result)

    async def aresults(self) -> t.List[t.Any]:
        """
        Execute all submitted jobs and return their results asynchronously.
        The results are returned in the order of job submission.

        This is the async entry point for executing async jobs when already in an async context.
        """
        results = await self._process_jobs()
        # If raise_exceptions is True, propagate the exception
        for r in results:
            if self.raise_exceptions and isinstance(r, Exception):
                raise r
        sorted_results = sorted(results, key=lambda x: x[0])
        return [r[1] for r in sorted_results]

    def results(self) -> t.List[t.Any]:
        """
        Execute all submitted jobs and return their results. The results are returned in the order of job submission.

        This is the main sync entry point for executing async jobs.
        """

        async def _async_wrapper():
            return await self.aresults()

        apply_nest_asyncio()
        return run(_async_wrapper)


def run_async_batch(
    desc: str,
    func: t.Callable,
    kwargs_list: t.List[t.Dict],
    batch_size: t.Optional[int] = None,
):
    """
    Provide functionality to run the same async function with different arguments in parallel.
    """
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
