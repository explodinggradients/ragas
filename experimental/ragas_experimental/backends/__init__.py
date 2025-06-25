# Optional imports for backends that require additional dependencies

# Always available backends
from .ragas_api_client import RagasApiClient
from .factory import RagasApiClientFactory

# Conditionally import Google Drive backend
try:
    from .gdrive_backend import GDriveBackend
    __all__ = ["RagasApiClient", "RagasApiClientFactory", "GDriveBackend"]
except ImportError:
    __all__ = ["RagasApiClient", "RagasApiClientFactory"]

# Conditionally import Notion backend if available
try:
    from .notion_backend import NotionBackend
    __all__.append("NotionBackend")
except ImportError:
    pass
