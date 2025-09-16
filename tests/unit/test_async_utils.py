import asyncio

import pytest

from ragas.async_utils import run_async_tasks


def test_is_event_loop_running_in_script():
    from ragas.async_utils import is_event_loop_running

    assert is_event_loop_running() is False


def test_as_completed_in_script():
    from ragas.async_utils import as_completed

    async def echo_order(index: int, delay: float):
        await asyncio.sleep(delay)
        return index

    async def _run():
        # Use decreasing delays so results come out in reverse order
        coros = [echo_order(1, 0.3), echo_order(2, 0.2), echo_order(3, 0.1)]
        results = []
        for t in as_completed(coros, 3):
            r = await t
            results.append(r)
        return results

    results = asyncio.run(_run())
    # Results should be [3, 2, 1] due to decreasing delays
    assert results == [3, 2, 1]


def test_as_completed_max_workers():
    import time

    from ragas.async_utils import as_completed

    async def sleeper(idx):
        await asyncio.sleep(0.1)
        return idx

    async def _run():
        start = time.time()
        coros = [sleeper(i) for i in range(5)]
        results = []
        for t in as_completed(coros, max_workers=2):
            r = await t
            results.append(r)
        elapsed = time.time() - start
        return results, elapsed

    results, elapsed = asyncio.run(_run())
    # With max_workers=2, total time should be at least 0.2s for 5 tasks
    assert len(results) == 5
    assert elapsed >= 0.2


def test_run_function():
    from ragas.async_utils import run

    async def foo():
        return 42

    result = run(foo)
    assert result == 42


@pytest.fixture
def tasks():
    async def echo_order(index: int):
        return index

    return [echo_order(i) for i in range(1, 11)]


def test_run_async_tasks_unbatched(tasks):
    results = run_async_tasks(tasks)
    assert sorted(results) == sorted(range(1, 11))


def test_run_async_tasks_batched(tasks):
    results = run_async_tasks(tasks, batch_size=3)
    assert sorted(results) == sorted(range(1, 11))


def test_run_async_tasks_no_progress(tasks):
    results = run_async_tasks(tasks, show_progress=False)
    assert sorted(results) == sorted(range(1, 11))
