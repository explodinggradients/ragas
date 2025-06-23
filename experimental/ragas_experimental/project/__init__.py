"""Project management module for Ragas experimental framework.

This module provides a clean interface for managing AI projects with support for
multiple backend storage options including local CSV files and the Ragas platform.
"""

from .core import Project
from .utils import MemorableNames, memorable_names, create_nano_id
from .backends import (
    ProjectBackend, 
    DatasetBackend, 
    create_project_backend,
    list_backends,
    print_available_backends,
    register_backend,
)

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
    name: str,
    description: str = "",
    backend: str = "local_csv",
    **kwargs
) -> Project:
    """Create a new project with the specified backend.
    
    Args:
        name: Name of the project
        description: Description of the project
        backend: Backend type ("local_csv" or "platform")
        **kwargs: Additional backend-specific arguments
        
    Returns:
        Project: A new project instance
        
    Examples:
        >>> # Create a local project
        >>> project = create_project("my_project", backend="local_csv", root_dir="/path/to/projects")
        
        >>> # Create a platform project  
        >>> project = create_project("my_project", backend="platform", ragas_api_client=client)
    """
    return Project.create(name=name, description=description, backend=backend, **kwargs)


def get_project(
    name: str,
    backend: str = "local_csv",
    **kwargs
) -> Project:
    """Get an existing project by name.
    
    Args:
        name: Name of the project to retrieve
        backend: Backend type ("local_csv" or "platform")
        **kwargs: Additional backend-specific arguments
        
    Returns:
        Project: The existing project instance
        
    Examples:
        >>> # Get a local project
        >>> project = get_project("my_project", backend="local_csv", root_dir="/path/to/projects")
        
        >>> # Get a platform project
        >>> project = get_project("my_project", backend="platform", ragas_api_client=client)
    """
    return Project.get(name=name, backend=backend, **kwargs)