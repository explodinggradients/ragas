"""Async utils."""

import asyncio
import typing as t

from tqdm.asyncio import tqdm as atqdm

from ragas.executor import is_event_loop_running
from ragas.utils import batched


def run_async_tasks(
    tasks: t.List[t.Coroutine],
    batch_size: t.Optional[int] = None,
    show_progress: bool = True,
    progress_bar_desc: str = "Running async tasks",
) -> t.List[t.Any]:
    """Run a list of async tasks."""

    async def _process_batch(
        batch: t.Iterable[t.Coroutine],
        progress_bar_desc: str = "Batch progress",
    ) -> t.List[t.Tuple[int, t.Coroutine]]:
        """Process batch of jobs with optional progress tracking."""

        return await atqdm.gather(
            *list(batch),
            desc=progress_bar_desc,
            disable=not show_progress,
        )

    async def _process_jobs() -> t.List[t.Any]:
        """Execute jobs with optional progress tracking."""

        if batch_size:
            results = []
            batches = batched(tasks, batch_size)  # generator
            n_batches = (len(tasks) + batch_size - 1) // batch_size

            with atqdm(
                total=n_batches,
                desc=progress_bar_desc,
                disable=not show_progress,
            ) as pbar:
                for i, batch in enumerate(batches, 1):
                    batch_results = await _process_batch(
                        batch,
                        progress_bar_desc=f"Batch {i}/{n_batches}",
                    )
                    results.extend(batch_results)
                    pbar.update(1)

        else:
            results = await _process_batch(tasks, progress_bar_desc=progress_bar_desc)

        return results

    if is_event_loop_running():
        # an event loop is running so call nested_asyncio to fix this
        try:
            import nest_asyncio
        except ImportError:
            raise ImportError(
                "It seems like your running this in a jupyter-like environment. "
                "Please install nest_asyncio with `pip install nest_asyncio` to make it work."
            )
        finally:
            nest_asyncio.apply()

    results = asyncio.run(_process_jobs())
    return results
