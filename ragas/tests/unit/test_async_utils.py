import asyncio

import pytest

from ragas.async_utils import as_completed, is_event_loop_running, run_async_tasks


def test_is_event_loop_running_in_script():
    assert is_event_loop_running() is False


def test_as_completed_in_script():
    async def echo_order(index: int):
        await asyncio.sleep(index)
        return index

    async def _run():
        results = []
        for t in await as_completed([echo_order(1), echo_order(2), echo_order(3)], 3):
            r = await t
            results.append(r)
        return results

    results = asyncio.run(_run())

    assert results == [1, 2, 3]


@pytest.fixture
def tasks():
    async def echo_order(index: int):
        return index

    return [echo_order(i) for i in range(1, 11)]


@pytest.mark.asyncio
async def test_run_async_tasks_unbatched(tasks):
    # Act
    results = run_async_tasks(tasks)

    # Assert
    assert sorted(results) == sorted(range(1, 11))


@pytest.mark.asyncio
async def test_run_async_tasks_batched(tasks):
    # Act
    results = run_async_tasks(tasks, batch_size=3)

    # Assert
    assert sorted(results) == sorted(range(1, 11))


@pytest.mark.asyncio
async def test_run_async_tasks_no_progress(tasks):
    # Act
    results = run_async_tasks(tasks, show_progress=False)

    # Assert
    assert sorted(results) == sorted(range(1, 11))
