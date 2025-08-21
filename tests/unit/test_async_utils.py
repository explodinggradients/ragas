import pytest

from ragas.async_utils import run_async_tasks


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
