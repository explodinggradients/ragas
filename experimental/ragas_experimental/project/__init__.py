"""Project management module for Ragas experimental framework.

This module provides a clean interface for managing AI projects with support for
multiple backend storage options including local CSV files and the Ragas app.
"""

from .backends import (
    DatasetBackend,
    ProjectBackend,
    create_project_backend,
    list_backends,
    print_available_backends,
    register_backend,
)
from .core import Project
from .utils import MemorableNames, create_nano_id, memorable_names

__all__ = [
    "Project",
    "create_project",
    "get_project",
    "MemorableNames",
    "memorable_names",
    "create_nano_id",
    "ProjectBackend",
    "DatasetBackend",
    "create_project_backend",
    "list_backends",
    "print_available_backends",
    "register_backend",
]


def create_project(
    name: str, description: str = "", backend: str = "local/csv", **kwargs
) -> Project:
    """Create a new project with the specified backend.

    Args:
        name: Name of the project
        description: Description of the project
        backend: Backend type ("local/csv" or "ragas/app")
        **kwargs: Additional backend-specific arguments

    Returns:
        Project: A new project instance

    Examples:
        >>> # Create a local project
        >>> project = create_project("my_project", backend="local/csv", root_dir="/path/to/projects")

        >>> # Create a ragas/app project
        >>> project = create_project("my_project", backend="ragas/app", ragas_api_client=client)
    """
    return Project.create(name=name, description=description, backend=backend, **kwargs)


def get_project(name: str, backend: str = "local/csv", **kwargs) -> Project:
    """Get an existing project by name.

    Args:
        name: Name of the project to retrieve
        backend: Backend type ("local/csv" or "ragas/app")
        **kwargs: Additional backend-specific arguments

    Returns:
        Project: The existing project instance

    Examples:
        >>> # Get a local project
        >>> project = get_project("my_project", backend="local/csv", root_dir="/path/to/projects")

        >>> # Get a ragas/app project
        >>> project = get_project("my_project", backend="ragas/app", ragas_api_client=client)
    """
    return Project.get(name=name, backend=backend, **kwargs)
