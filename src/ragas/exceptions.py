from __future__ import annotations


class RagasException(Exception):
    """
    Base exception class for ragas.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class OpenAIKeyNotFound(RagasException):
    message: str = "OpenAI API key not found! Seems like your trying to use Ragas metrics with OpenAI endpoints. Please set 'OPENAI_API_KEY' environment variable"  # noqa

    def __init__(self):
        super().__init__(self.message)


class AzureOpenAIKeyNotFound(RagasException):
    message: str = "AzureOpenAI API key not found! Seems like your trying to use Ragas metrics with AzureOpenAI endpoints. Please set 'AZURE_OPENAI_API_KEY' environment variable"  # noqa

    def __init__(self):
        super().__init__(self.message)
