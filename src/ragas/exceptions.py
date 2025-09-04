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

    def __init__(self):
        msg = "The output parser failed to parse the output including retries."
        super().__init__(msg)


class LLMDidNotFinishException(RagasException):
    """
    Exception raised when the LLM did not finish.
    """

    def __init__(self):
        msg = "The LLM generation was not completed. Please increase the max_tokens and try again."
        super().__init__(msg)


# Exceptions migrated from experimental module
class RagasError(Exception):
    """Base class for all Ragas-related exceptions."""

    pass


class ValidationError(RagasError):
    """Raised when field validation fails."""

    pass


class DuplicateError(RagasError):
    """Exception raised when a duplicate resource is created."""

    pass


class NotFoundError(RagasError):
    """Exception raised when a resource is not found."""

    pass


class ResourceNotFoundError(NotFoundError):
    """Exception raised when a resource doesn't exist."""

    pass


class ProjectNotFoundError(ResourceNotFoundError):
    """Exception raised when a project doesn't exist."""

    pass


class DatasetNotFoundError(ResourceNotFoundError):
    """Exception raised when a dataset doesn't exist."""

    pass


class ExperimentNotFoundError(ResourceNotFoundError):
    """Exception raised when an experiment doesn't exist."""

    pass


class DuplicateResourceError(RagasError):
    """Exception raised when multiple resources exist with the same identifier."""

    pass


class DuplicateProjectError(DuplicateResourceError):
    """Exception raised when multiple projects exist with the same name."""

    pass


class DuplicateDatasetError(DuplicateResourceError):
    """Exception raised when multiple datasets exist with the same name."""

    pass


class DuplicateExperimentError(DuplicateResourceError):
    """Exception raised when multiple experiments exist with the same name."""

    pass
