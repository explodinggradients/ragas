from __future__ import annotations


class RagasException(Exception):
    """
    Base exception class for ragas.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)
