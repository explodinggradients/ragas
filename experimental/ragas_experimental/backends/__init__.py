"""Backend factory and exports for all backends."""

from .base import BaseBackend
from .registry import (
    BackendRegistry,
    BACKEND_REGISTRY,
    get_registry,
    print_available_backends,
    register_backend,
)

# concrete backends
from .local_csv import LocalCSVBackend
from .local_jsonl import LocalJSONLBackend

# Optional backends that require additional dependencies
try:
    from .notion import NotionBackend
    NOTION_AVAILABLE = True
except ImportError:
    NotionBackend = None
    NOTION_AVAILABLE = False


__all__ = [
    "BaseBackend",
    "BackendRegistry",
    "LocalCSVBackend",
    "LocalJSONLBackend",
    "get_registry",
    "register_backend",
    "print_available_backends",
    "BACKEND_REGISTRY",
]

if NOTION_AVAILABLE:
    __all__.append("NotionBackend")
