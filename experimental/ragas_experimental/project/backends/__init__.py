"""Backend factory and exports for project backends."""

from .base import ProjectBackend, DatasetBackend
from .local_csv import LocalCSVProjectBackend
from .platform import PlatformProjectBackend

__all__ = [
    "ProjectBackend",
    "DatasetBackend", 
    "LocalCSVProjectBackend",
    "PlatformProjectBackend",
    "create_project_backend",
]


def create_project_backend(backend_type: str, **kwargs) -> ProjectBackend:
    """Factory function to create the appropriate project backend.
    
    Args:
        backend_type: The type of backend to create ("local_csv" or "platform")
        **kwargs: Arguments specific to the backend
        
    Returns:
        ProjectBackend: An instance of the requested backend
    """
    backend_classes = {
        "local_csv": LocalCSVProjectBackend,
        "local": LocalCSVProjectBackend,  # Backward compatibility
        "platform": PlatformProjectBackend,
        "ragas_app": PlatformProjectBackend,  # Backward compatibility
    }
    
    if backend_type not in backend_classes:
        raise ValueError(f"Unsupported backend: {backend_type}")
        
    return backend_classes[backend_type](**kwargs)