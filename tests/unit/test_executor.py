import asyncio

import pytest

from ragas.executor import Executor


def test_is_event_loop_running_in_script():
    from ragas.executor import is_event_loop_running

    assert is_event_loop_running() is False


def test_as_completed_in_script():
    from ragas.executor import as_completed

    async def echo_order(index: int):
        await asyncio.sleep(0.1)
        return index

    async def _run():
        results = []
        for t in await as_completed([echo_order(1), echo_order(2), echo_order(3)], 3):
            r = await t
            results.append(r)
        return results

    results = asyncio.run(_run())

    assert results == [1, 2, 3]


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [None, 3, 20])
async def test_order_of_execution(batch_size):
    async def echo_order(index: int):
        await asyncio.sleep(1 / index)
        return index

    # Arrange
    executor = Executor(batch_size=batch_size)
    # add 10 jobs to the executor
    for i in range(1, 11):
        executor.submit(echo_order, i, name=f"echo_order_{i}")

    # Act
    results = executor.results()
    # Assert
    assert results == list(range(1, 11))


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [None, 3, 20])
async def test_executor_in_script(batch_size):
    async def echo_order(index: int):
        await asyncio.sleep(1 / index)
        return index

    # Arrange
    executor = Executor(batch_size=batch_size)
    # add 10 jobs to the executor
    for i in range(1, 4):
        executor.submit(echo_order, i, name=f"echo_order_{i}")

    # Act
    results = executor.results()
    # Assert
    assert results == list(range(1, 4))


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [None, 3, 20])
async def test_executor_with_running_loop(batch_size):
    loop = asyncio.new_event_loop()
    loop.run_until_complete(asyncio.sleep(0.1))

    async def echo_order(index: int):
        await asyncio.sleep(1 / index)
        return index

    # Arrange
    executor = Executor(batch_size=batch_size)
    for i in range(1, 4):
        executor.submit(echo_order, i, name=f"echo_order_{i}")

    # Act
    # add 10 jobs to the executor
    results = executor.results()
    # Assert
    assert results == list(range(1, 4))
