"""Async utils."""

import asyncio
import logging
import typing as t

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


def apply_nest_asyncio():
    """Apply nest_asyncio if an event loop is running."""
    if is_event_loop_running():
        # an event loop is running so call nested_asyncio to fix this
        try:
            import nest_asyncio
        except ImportError:
            raise ImportError(
                "It seems like your running this in a jupyter-like environment. Please install nest_asyncio with `pip install nest_asyncio` to make it work."
            )

        nest_asyncio.apply()


def as_completed(
    coroutines: t.Sequence[t.Coroutine],
    max_workers: int = -1,
    *,
    cancel_check: t.Optional[t.Callable[[], bool]] = None,
    cancel_pending: bool = True,
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

    ac_iter = asyncio.as_completed(tasks)

    if cancel_check is None:
        return ac_iter

    def _iter_with_cancel():
        for future in ac_iter:
            if cancel_check():
                if cancel_pending:
                    for t in tasks:
                        if not t.done():
                            t.cancel()
                break
            yield future

    return _iter_with_cancel()


async def process_futures(
    futures: t.Iterator[asyncio.Future],
) -> t.AsyncGenerator[t.Any, None]:
    """
    Process futures with optional progress tracking.

    Args:
        futures: Iterator of asyncio futures to process (e.g., from asyncio.as_completed)

    Yields:
        Results from completed futures as they finish
    """
    # Process completed futures as they finish
    for future in futures:
        try:
            result = await future
        except asyncio.CancelledError:
            raise  # Re-raise CancelledError to ensure proper cancellation
        except Exception as e:
            result = e

        yield result


def run(
    async_func: t.Union[
        t.Callable[[], t.Coroutine[t.Any, t.Any, t.Any]],
        t.Coroutine[t.Any, t.Any, t.Any],
    ],
    allow_nest_asyncio: bool = True,
) -> t.Any:
    """
    Run an async function in the current event loop or a new one if not running.

    Parameters
    ----------
    async_func : Callable or Coroutine
        The async function or coroutine to run
    allow_nest_asyncio : bool, optional
        Whether to apply nest_asyncio for Jupyter compatibility. Default is True.
        Set to False in production environments to avoid event loop patching.
    """
    # Only apply nest_asyncio if explicitly allowed
    if allow_nest_asyncio:
        apply_nest_asyncio()

    # Create the coroutine if it's a callable, otherwise use directly
    coro = async_func() if callable(async_func) else async_func
    return asyncio.run(coro)


def run_async_tasks(
    tasks: t.Sequence[t.Coroutine],
    batch_size: t.Optional[int] = None,
    show_progress: bool = True,
    progress_bar_desc: str = "Running async tasks",
    max_workers: int = -1,
    *,
    cancel_check: t.Optional[t.Callable[[], bool]] = None,
) -> t.List[t.Any]:
    """
    Execute async tasks with optional batching and progress tracking.

    NOTE: Order of results is not guaranteed!

    Args:
        tasks: Sequence of coroutines to execute
        batch_size: Optional size for batching tasks. If None, runs all concurrently
        show_progress: Whether to display progress bars
        max_workers: Maximum number of concurrent tasks (-1 for unlimited)
    """
    from ragas.utils import ProgressBarManager, batched

    async def _run():
        total_tasks = len(tasks)
        results = []
        pbm = ProgressBarManager(progress_bar_desc, show_progress)

        if not batch_size:
            with pbm.create_single_bar(total_tasks) as pbar:
                async for result in process_futures(
                    as_completed(tasks, max_workers, cancel_check=cancel_check)
                ):
                    results.append(result)
                    pbar.update(1)
        else:
            total_tasks = len(tasks)
            batches = batched(tasks, batch_size)
            overall_pbar, batch_pbar, n_batches = pbm.create_nested_bars(
                total_tasks, batch_size
            )
            with overall_pbar, batch_pbar:
                for i, batch in enumerate(batches, 1):
                    pbm.update_batch_bar(batch_pbar, i, n_batches, len(batch))
                    async for result in process_futures(
                        as_completed(batch, max_workers, cancel_check=cancel_check)
                    ):
                        results.append(result)
                        batch_pbar.update(1)
                    overall_pbar.update(len(batch))

        return results

    return run(_run)
