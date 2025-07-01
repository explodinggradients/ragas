"""Project management module for Ragas experimental framework.

This module provides a clean interface for managing AI projects with support for
multiple backend storage options including local CSV files and the Ragas app.
"""

from ..backends import (
    DataTableBackend,
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
    "MemorableNames",
    "memorable_names",
    "create_nano_id",
    "ProjectBackend",
    "DataTableBackend",
    "create_project_backend",
    "list_backends",
    "print_available_backends",
    "register_backend",
]
