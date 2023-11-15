"""Async utils."""
import asyncio
from typing import Any, Coroutine, List


def run_async_tasks(
    tasks: List[Coroutine],
    show_progress: bool = False,
    progress_bar_desc: str = "Running async tasks",
) -> List[Any]:
    """Run a list of async tasks."""
    tasks_to_execute: List[Any] = tasks

    # if running in notebook, use nest_asyncio to hijack the event loop
    try:
        loop = asyncio.get_running_loop()
        try:
            import nest_asyncio
        except ImportError:
            raise RuntimeError(
                "nest_asyncio is required to run async tasks in jupyter. Please install it via `pip install nest_asyncio`."  # noqa
            )
        else:
            nest_asyncio.apply()
    except RuntimeError:
        loop = asyncio.new_event_loop()

    # gather tasks to run
    if show_progress:
        from tqdm.asyncio import tqdm

        async def _gather() -> List[Any]:
            "gather tasks and show progress bar"
            return await tqdm.gather(*tasks_to_execute, desc=progress_bar_desc)

    else:  # don't show_progress

        async def _gather() -> List[Any]:
            return await asyncio.gather(*tasks_to_execute)

    try:
        outputs: List[Any] = loop.run_until_complete(_gather())
    except Exception as e:
        # run the operation w/o tqdm on hitting a fatal
        # may occur in some environments where tqdm.asyncio
        # is not supported
        raise RuntimeError("Fatal error occurred while running async tasks.", e)
    return outputs
