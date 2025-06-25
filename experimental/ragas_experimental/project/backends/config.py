"""Configuration classes for all backend types."""

from abc import ABC
from typing import Optional, Literal
from ragas_experimental.model.pydantic_model import (
    ExtendedPydanticBaseModel as BaseModel,
)
from pydantic import ConfigDict


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
    Requires Box application setup and authentication credentials.
    """

    # Authentication
    auth_type: Literal["jwt", "oauth2"] = "jwt"
    """Authentication method. 'jwt' for enterprise apps, 'oauth2' for user apps."""

    client_id: str
    """Box application client ID from Box Developer Console."""

    client_secret: str
    """Box application client secret from Box Developer Console."""

    # JWT-specific fields
    enterprise_id: Optional[str] = None
    """Enterprise ID for JWT authentication. Required for JWT auth."""

    jwt_key_id: Optional[str] = None
    """JWT key ID from Box application configuration. Required for JWT auth."""

    private_key_path: Optional[str] = None
    """Path to private key PEM file. Alternative to private_key."""

    private_key: Optional[str] = None
    """Private key content as string. Alternative to private_key_path."""

    private_key_passphrase: Optional[str] = None
    """Passphrase for encrypted private key. Optional."""

    # OAuth2-specific fields
    access_token: Optional[str] = None
    """User access token for OAuth2 authentication. Required for OAuth2 auth."""

    refresh_token: Optional[str] = None
    """Refresh token for OAuth2 token renewal. Optional but recommended."""

    # Optional settings
    root_folder_id: str = "0"
    """Box folder ID to use as root. '0' is Box root folder."""

    model_config = ConfigDict(env_prefix="BOX_")

    def model_post_init(self, __context):
        """Validate configuration after initialization."""
        if self.auth_type == "jwt":
            if not all([self.enterprise_id, self.jwt_key_id]):
                raise ValueError("JWT auth requires enterprise_id and jwt_key_id")
            if not (self.private_key_path or self.private_key):
                raise ValueError(
                    "JWT auth requires either private_key_path or private_key"
                )
        elif self.auth_type == "oauth2":
            if not self.access_token:
                raise ValueError("OAuth2 auth requires access_token")


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
