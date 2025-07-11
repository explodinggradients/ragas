"""Backend factory and exports for all backends."""

from .base import BaseBackend
from .registry import (
    BackendRegistry,
    create_backend,
    get_backend_info,
    get_registry,
    list_backend_info,
    list_backends,
    print_available_backends,
    register_backend,
)

# concrete backends
from .local_csv import LocalCSVBackend
from .local_jsonl import LocalJSONLBackend


__all__ = [
    "BaseBackend",
    "BackendRegistry",
    "LocalCSVBackend",
    "LocalJSONLBackend",
    "get_registry",
    "register_backend",
    "list_backends",
    "get_backend_info",
    "list_backend_info",
    "print_available_backends",
    "create_backend",
]
