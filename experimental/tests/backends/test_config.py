"""Tests for backend configuration classes."""

import pytest
from ragas_experimental.project.backends.config import (
    LocalCSVConfig,
    RagasAppConfig,
)

# Import BoxCSVConfig if available
try:
    from ragas_experimental.project.backends.config import BoxCSVConfig
    HAS_BOX_CONFIG = True
except ImportError:
    HAS_BOX_CONFIG = False


def test_local_csv_config():
    """Test LocalCSV configuration."""
    config = LocalCSVConfig(root_dir="/custom/path")
    assert config.root_dir == "/custom/path"
    
    # Test defaults
    default_config = LocalCSVConfig()
    assert default_config.root_dir == "./ragas_data"


@pytest.mark.skipif(not HAS_BOX_CONFIG, reason="Box SDK not available")
def test_box_csv_config_jwt():
    """Test Box CSV configuration with JWT auth."""
    config = BoxCSVConfig(
        auth_type="jwt",
        client_id="test_client",
        client_secret="test_secret",
        enterprise_id="test_enterprise",
        jwt_key_id="test_key_id",
        private_key="test_private_key"
    )
    
    assert config.auth_type == "jwt"
    assert config.client_id == "test_client"
    assert config.client_secret == "test_secret"
    assert config.enterprise_id == "test_enterprise"
    assert config.jwt_key_id == "test_key_id"
    assert config.private_key == "test_private_key"


@pytest.mark.skipif(not HAS_BOX_CONFIG, reason="Box SDK not available")
def test_box_csv_config_oauth2():
    """Test Box CSV configuration with OAuth2 auth."""
    config = BoxCSVConfig(
        auth_type="oauth2",
        client_id="test_client",
        client_secret="test_secret",
        access_token="test_access_token"
    )
    
    assert config.auth_type == "oauth2"
    assert config.client_id == "test_client"
    assert config.client_secret == "test_secret"
    assert config.access_token == "test_access_token"


@pytest.mark.skipif(not HAS_BOX_CONFIG, reason="Box SDK not available")
def test_box_csv_config_validation_jwt_missing_fields():
    """Test Box CSV configuration validation for JWT with missing fields."""
    # Should raise error for JWT without required fields
    with pytest.raises(ValueError, match="JWT auth requires enterprise_id and jwt_key_id"):
        BoxCSVConfig(
            auth_type="jwt",
            client_id="test_client",
            client_secret="test_secret"
            # Missing enterprise_id, jwt_key_id, private_key
        )


@pytest.mark.skipif(not HAS_BOX_CONFIG, reason="Box SDK not available")
def test_box_csv_config_validation_jwt_missing_private_key():
    """Test Box CSV configuration validation for JWT with missing private key."""
    with pytest.raises(ValueError, match="JWT auth requires either private_key_path or private_key"):
        BoxCSVConfig(
            auth_type="jwt",
            client_id="test_client",
            client_secret="test_secret",
            enterprise_id="test_enterprise",
            jwt_key_id="test_key_id"
            # Missing private_key or private_key_path
        )


@pytest.mark.skipif(not HAS_BOX_CONFIG, reason="Box SDK not available")
def test_box_csv_config_validation_oauth2_missing_token():
    """Test Box CSV configuration validation for OAuth2 with missing access token."""
    with pytest.raises(ValueError, match="OAuth2 auth requires access_token"):
        BoxCSVConfig(
            auth_type="oauth2",
            client_id="test_client",
            client_secret="test_secret"
            # Missing access_token
        )


@pytest.mark.skipif(not HAS_BOX_CONFIG, reason="Box SDK not available")
def test_box_csv_config_defaults():
    """Test Box CSV configuration defaults."""
    config = BoxCSVConfig(
        client_id="test_client",
        client_secret="test_secret",
        enterprise_id="test_enterprise",
        jwt_key_id="test_key_id",
        private_key="test_private_key"
    )
    
    # Test defaults
    assert config.auth_type == "jwt"  # default
    assert config.root_folder_id == "0"  # default


def test_ragas_app_config():
    """Test Ragas App configuration."""
    config = RagasAppConfig(api_key="test_key")
    assert config.api_key == "test_key"
    assert config.api_url == "https://api.ragas.io"  # default
    assert config.timeout == 30  # default
    assert config.max_retries == 3  # default


def test_ragas_app_config_custom_values():
    """Test Ragas App configuration with custom values."""
    config = RagasAppConfig(
        api_url="https://custom.api.com",
        api_key="custom_key",
        timeout=60,
        max_retries=5
    )
    
    assert config.api_url == "https://custom.api.com"
    assert config.api_key == "custom_key"
    assert config.timeout == 60
    assert config.max_retries == 5


def test_ragas_app_config_defaults():
    """Test Ragas App configuration defaults."""
    config = RagasAppConfig()
    
    assert config.api_url == "https://api.ragas.io"
    assert config.api_key is None
    assert config.timeout == 30
    assert config.max_retries == 3