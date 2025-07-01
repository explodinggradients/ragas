"""Backend factory and exports for all backends."""

from .base import DataTableBackend, ProjectBackend

# Import concrete backends
from .local_csv import LocalCSVProjectBackend
from .ragas_app import RagasAppProjectBackend

# Optional backends with dependencies
try:
    from .box_csv import BoxCSVProjectBackend
except ImportError:
    BoxCSVProjectBackend = None

# Import configuration classes
from .config import BackendConfig, LocalCSVConfig, RagasAppConfig

try:
    from .config import BoxCSVConfig
except ImportError:
    BoxCSVConfig = None

from .registry import (
    BackendRegistry,
    create_project_backend,
    get_backend_info,
    get_registry,
    list_backend_info,
    list_backends,
    print_available_backends,
    register_backend,
)

# Import API client
from .ragas_api_client import RagasApiClient

__all__ = [
    "ProjectBackend",
    "DataTableBackend",
    "BackendRegistry",
    "get_registry",
    "register_backend",
    "list_backends",
    "get_backend_info",
    "list_backend_info",
    "print_available_backends",
    "create_project_backend",
    # Configuration classes
    "BackendConfig",
    "LocalCSVConfig",
    "RagasAppConfig",
    "BoxCSVConfig",
    # Concrete backends
    "LocalCSVProjectBackend",
    "RagasAppProjectBackend",
    "BoxCSVProjectBackend",
    # API client
    "RagasApiClient",
]
