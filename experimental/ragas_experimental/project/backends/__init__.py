"""Backend factory and exports for project backends."""

from .base import ProjectBackend, DatasetBackend
from .registry import (
    BackendRegistry,
    get_registry,
    register_backend,
    list_backends,
    get_backend_info,
    list_backend_info,
    print_available_backends,
    create_project_backend,
)

# Import concrete backends for backward compatibility
from .local_csv import LocalCSVProjectBackend
from .platform import PlatformProjectBackend

__all__ = [
    "ProjectBackend",
    "DatasetBackend",
    "BackendRegistry", 
    "get_registry",
    "register_backend",
    "list_backends",
    "get_backend_info",
    "list_backend_info", 
    "print_available_backends",
    "create_project_backend",
    # Concrete backends for backward compatibility
    "LocalCSVProjectBackend",
    "PlatformProjectBackend",
]