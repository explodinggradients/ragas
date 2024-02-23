from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from ragas.testset.evolutions import Evolution


class RagasException(Exception):
    """
    Base exception class for ragas.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class MaxRetriesExceeded(RagasException):
    """
    Exception raised when the maximum number of retries is exceeded.
    """

    def __init__(self, evolution: Evolution):
        self.evolution = evolution
        msg = f"Max retries exceeded for evolution {evolution.__class__.__name__}."
        super().__init__(msg)


class ExceptionInRunner(RagasException):
    """
    Exception raised when an exception is raised in the executor.
    """

    def __init__(self):
        msg = "The runner thread which was running the jobs raised an exeception. Read the traceback above to debug it. You can also pass `raise_exceptions=False` incase you want to show only a warning message instead."
        super().__init__(msg)
