import asyncio

import pytest


@pytest.mark.asyncio
async def test_order_of_execution():
    from ragas.executor import Executor

    async def echo_order(index: int):
        return index

    # Arrange
    executor = Executor()

    # Act
    # add 10 jobs to the executor
    for i in range(1, 11):
        executor.submit(echo_order, i, name=f"echo_order_{i}")
    results = executor.results()
    # Assert
    assert results == list(range(1, 11))


@pytest.mark.asyncio
async def test_executor_in_script():
    from ragas.executor import Executor

    async def echo_order(index: int):
        await asyncio.sleep(0.1)
        return index

    # Arrange
    executor = Executor()

    # Act
    # add 10 jobs to the executor
    for i in range(1, 4):
        executor.submit(echo_order, i, name=f"echo_order_{i}")
    results = executor.results()
    # Assert
    assert results == list(range(1, 4))


@pytest.mark.asyncio
async def test_executor_with_running_loop():
    import asyncio

    from ragas.executor import Executor

    loop = asyncio.new_event_loop()
    loop.run_until_complete(asyncio.sleep(0.1))

    async def echo_order(index: int):
        await asyncio.sleep(0.1)
        return index

    # Arrange
    executor = Executor()

    # Act
    # add 10 jobs to the executor
    for i in range(1, 4):
        executor.submit(echo_order, i, name=f"echo_order_{i}")
    results = executor.results()
    # Assert
    assert results == list(range(1, 4))


def test_is_event_loop_running_in_script():
    from ragas.executor import is_event_loop_running

    assert is_event_loop_running() is False


def test_as_completed_in_script():
    from ragas.executor import as_completed

    async def echo_order(index: int):
        await asyncio.sleep(index)
        return index

    async def _run():
        results = []
        for t in as_completed([echo_order(1), echo_order(2), echo_order(3)], 3):
            r = await t
            results.append(r)
        return results

    results = asyncio.run(_run())

    assert results == [1, 2, 3]
