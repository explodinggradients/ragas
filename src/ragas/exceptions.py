from __future__ import annotations


class RagasException(Exception):
    """
    Base exception class for ragas.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class ExceptionInRunner(RagasException):
    """
    Exception raised when an exception is raised in the executor.
    """

    def __init__(self):
        msg = "The runner thread which was running the jobs raised an exeception. Read the traceback above to debug it. You can also pass `raise_exceptions=False` incase you want to show only a warning message instead."
        super().__init__(msg)


class RagasOutputParserException(RagasException):
    """
    Exception raised when the output parser fails to parse the output.
    """

    def __init__(self, num_retries: int):
        msg = (
            f"The output parser failed to parse the output after {num_retries} retries."
        )
        super().__init__(msg)
