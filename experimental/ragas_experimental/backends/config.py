"""Configuration classes for all backend types."""

from abc import ABC
from typing import Optional, Literal, TYPE_CHECKING, Any, Protocol, runtime_checkable, Annotated
from ragas_experimental.model.pydantic_model import (
    ExtendedPydanticBaseModel as BaseModel,
)
from pydantic import ConfigDict, field_validator, Field, PlainValidator

# Type-only imports for Box SDK
if TYPE_CHECKING:
    from boxsdk import Client
else:
    try:
        from boxsdk import Client
    except ImportError:
        Client = None


# Protocol definitions for Box SDK interfaces
@runtime_checkable
class BoxUserProtocol(Protocol):
    """Protocol for Box user objects."""
    name: str


@runtime_checkable
class BoxUserManagerProtocol(Protocol):
    """Protocol for Box user manager objects."""
    def get(self) -> BoxUserProtocol: ...


@runtime_checkable
class BoxItemProtocol(Protocol):
    """Protocol for Box items (files/folders) returned by get_items()."""
    type: str  # "file" or "folder"
    name: str
    id: str


@runtime_checkable
class BoxFileProtocol(Protocol):
    """Protocol for Box file objects."""
    def content(self) -> bytes: ...
    def update_contents_with_stream(self, stream) -> None: ...


@runtime_checkable
class BoxFolderProtocol(Protocol):
    """Protocol for Box folder objects."""
    object_id: str
    
    def get_items(self) -> list[BoxItemProtocol]: ...
    def create_subfolder(self, name: str) -> "BoxFolderProtocol": ...
    def upload_stream(self, stream, filename: str) -> BoxFileProtocol: ...


@runtime_checkable
class BoxClientProtocol(Protocol):
    """Protocol for Box client objects."""
    def user(self) -> BoxUserManagerProtocol: ...
    def folder(self, folder_id: str) -> BoxFolderProtocol: ...
    def file(self, file_id: str) -> BoxFileProtocol: ...


def validate_box_client(value: Any) -> Any:
    """Validate that the value implements the BoxClientProtocol interface."""
    if value is None:
        raise ValueError("Box client is required")
    
    # Check if the object implements the required interface
    if not isinstance(value, BoxClientProtocol):
        # For mocks and other objects, check if they have the required methods
        required_methods = ['user', 'folder', 'file']
        for method in required_methods:
            if not hasattr(value, method):
                raise ValueError(f"Client must have {method} method")
    
    return value


# Type alias for the validated Box client
BoxClientType = Annotated[Any, PlainValidator(validate_box_client)]


class BackendConfig(BaseModel, ABC):
    """Base configuration class for all backends."""

    model_config = ConfigDict(validate_assignment=True)


class LocalCSVConfig(BackendConfig):
    """Configuration for Local CSV backend.

    Stores data in local CSV files organized in folder structure:
    root_dir/project_id/datasets/dataset_name.csv
    root_dir/project_id/experiments/experiment_name.csv
    """

    root_dir: str = "./ragas_data"
    """Root directory for storing CSV files. Defaults to './ragas_data'."""


class BoxCSVConfig(BackendConfig):
    """Configuration for Box CSV backend.

    Stores CSV files on Box cloud storage with same organization as local CSV.
    Requires an authenticated Box client to be provided.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: BoxClientType
    """Authenticated Box client. User must provide this."""

    root_folder_id: str = "0"
    """Box folder ID to use as root. '0' is Box root folder."""

    def model_post_init(self, __context):
        """Validate configuration after initialization."""
        # Verify the client is properly authenticated by attempting to get user info
        try:
            user = self.client.user().get()
            # Store user info for logging/debugging if needed
            self._authenticated_user = user.name
        except Exception as e:
            raise ValueError(f"Box client authentication failed: {e}")


class RagasAppConfig(BackendConfig):
    """Configuration for Ragas App Platform backend.

    Connects to the official Ragas platform service for cloud-based storage.
    """

    api_url: str = "https://api.ragas.io"
    """Ragas API base URL. Defaults to production API."""

    api_key: Optional[str] = None
    """API key for authentication. Can be set via RAGAS_API_KEY environment variable."""

    timeout: int = 30
    """Request timeout in seconds. Defaults to 30."""

    max_retries: int = 3
    """Maximum number of retry attempts for failed requests. Defaults to 3."""

    model_config = ConfigDict(env_prefix="RAGAS_")
