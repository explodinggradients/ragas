def test_order_of_execution():
    from ragas.executor import Executor

    async def echo_order(index):
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
