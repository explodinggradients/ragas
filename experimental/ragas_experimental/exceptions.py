"""All the exceptions specific to the `ragas_experimental` project."""

__all__ = [
    "RagasError",
    "ValidationError",
    "DuplicateError",
    "NotFoundError",
    "ResourceNotFoundError",
    "ProjectNotFoundError",
    "DatasetNotFoundError",
    "ExperimentNotFoundError",
    "DuplicateResourceError",
    "DuplicateProjectError",
    "DuplicateDatasetError",
    "DuplicateExperimentError",
]


class RagasError(Exception):
    """Base class for all Ragas-related exceptions."""

    pass


class ValidationError(RagasError):
    """Raised when field validation fails."""

    pass


class DuplicateError(RagasError):
    """Raised when multiple items are found but only one was expected."""

    pass


class NotFoundError(RagasError):
    """Raised when an item is not found."""

    pass


class ResourceNotFoundError(RagasError):
    """Exception raised when a requested resource doesn't exist."""

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
