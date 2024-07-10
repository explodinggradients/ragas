from asyncio import sleep


def test_order_of_execution():
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


def test_executor_in_script():
    from ragas.executor import Executor

    async def echo_order(index: int):
        await sleep(index)
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


def test_executor_with_running_loop():
    import asyncio
    from ragas.executor import Executor

    loop = asyncio.new_event_loop()

    async def echo_order(index: int):
        await sleep(1)
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
