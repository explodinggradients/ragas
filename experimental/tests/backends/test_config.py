"""Tests for backend configuration classes."""

import pytest
from ragas_experimental.backends.config import (
    LocalCSVConfig,
    RagasAppConfig,
)

# Import BoxCSVConfig if available
try:
    from ragas_experimental.backends.config import BoxCSVConfig
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
def test_box_csv_config_with_client():
    """Test Box CSV configuration with authenticated client."""
    from unittest.mock import MagicMock
    
    # Mock an authenticated client
    mock_client = MagicMock()
    mock_user = MagicMock()
    mock_user.name = "Test User"
    mock_client.user().get.return_value = mock_user
    
    config = BoxCSVConfig(client=mock_client)
    
    assert config.client == mock_client
    assert config.root_folder_id == "0"  # default
    assert config._authenticated_user == "Test User"


@pytest.mark.skipif(not HAS_BOX_CONFIG, reason="Box SDK not available")
def test_box_csv_config_with_custom_folder():
    """Test Box CSV configuration with custom root folder."""
    from unittest.mock import MagicMock
    
    mock_client = MagicMock()
    mock_user = MagicMock()
    mock_user.name = "Test User"
    mock_client.user().get.return_value = mock_user
    
    config = BoxCSVConfig(client=mock_client, root_folder_id="123456")
    
    assert config.client == mock_client
    assert config.root_folder_id == "123456"


@pytest.mark.skipif(not HAS_BOX_CONFIG, reason="Box SDK not available")
def test_box_csv_config_validation_missing_client():
    """Test Box CSV configuration validation for missing client."""
    from pydantic import ValidationError
    
    # Should raise error when client is not provided
    with pytest.raises(ValidationError):
        BoxCSVConfig()


@pytest.mark.skipif(not HAS_BOX_CONFIG, reason="Box SDK not available")
def test_box_csv_config_validation_invalid_client():
    """Test Box CSV configuration validation for invalid client."""
    from unittest.mock import MagicMock
    
    # Mock client that fails authentication
    mock_client = MagicMock()
    mock_client.user().get.side_effect = Exception("Authentication failed")
    
    with pytest.raises(ValueError, match="Box client authentication failed"):
        BoxCSVConfig(client=mock_client)


@pytest.mark.skipif(not HAS_BOX_CONFIG, reason="Box SDK not available")
def test_box_csv_config_none_client():
    """Test Box CSV configuration with None client."""
    from pydantic import ValidationError
    
    with pytest.raises(ValidationError):
        BoxCSVConfig(client=None)


@pytest.mark.skipif(not HAS_BOX_CONFIG, reason="Box SDK not available")
def test_box_csv_config_defaults():
    """Test Box CSV configuration defaults."""
    from unittest.mock import MagicMock
    
    mock_client = MagicMock()
    mock_user = MagicMock()
    mock_user.name = "Test User"
    mock_client.user().get.return_value = mock_user
    
    config = BoxCSVConfig(client=mock_client)
    
    # Should default to root folder "0"
    assert config.root_folder_id == "0"


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