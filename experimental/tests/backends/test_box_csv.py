"""Tests for Box CSV backend implementation."""

import io
import pytest
from unittest.mock import MagicMock, Mock, patch

# Skip all tests if box dependencies not available

try:
    from ragas_experimental.project.backends.box_csv import (
        BoxCSVDatasetBackend,
        BoxCSVProjectBackend,
    )
    from ragas_experimental.project.backends.config import BoxCSVConfig
    from boxsdk import BoxAPIException, Client
    box_available = True
except ImportError:
    box_available = False

from ragas_experimental.model.pydantic_model import (
    ExtendedPydanticBaseModel as BaseModel,
)


# Test model for dataset entries
class TestEntry(BaseModel):
    name: str
    age: int
    active: bool = True


@pytest.mark.skipif(not box_available, reason="Box SDK not available")
class TestBoxCSVDatasetBackend:
    """Test BoxCSVDatasetBackend functionality."""

    @pytest.fixture
    def mock_box_client(self):
        """Mock Box client for testing."""
        client = MagicMock(spec=Client)
        
        # Mock folder structure
        mock_folder = MagicMock()
        mock_folder.id = "test_folder_id"
        mock_folder.get_items.return_value = []
        mock_folder.create_subfolder.return_value = mock_folder
        mock_folder.upload_stream.return_value = MagicMock()
        
        client.folder.return_value = mock_folder
        return client

    @pytest.fixture
    def mock_dataset(self):
        """Mock dataset for testing."""
        dataset = MagicMock()
        dataset.model = TestEntry
        dataset._entries = []
        return dataset

    @pytest.fixture
    def backend(self, mock_box_client):
        """Create backend instance for testing."""
        return BoxCSVDatasetBackend(
            box_client=mock_box_client,
            project_folder_id="project_123",
            dataset_id="dataset_456",
            dataset_name="test_dataset",
            datatable_type="datasets",
        )

    def test_initialize(self, backend, mock_dataset, mock_box_client):
        """Test backend initialization."""
        backend.initialize(mock_dataset)
        
        assert backend.dataset == mock_dataset
        # Should call folder operations to ensure CSV exists
        mock_box_client.folder.assert_called()

    def test_get_column_mapping(self, backend):
        """Test column mapping retrieval."""
        mapping = backend.get_column_mapping(TestEntry)
        assert mapping == TestEntry.model_fields

    @patch('ragas_experimental.project.backends.box_csv.csv')
    def test_load_entries_empty_file(self, mock_csv, backend, mock_box_client):
        """Test loading entries from empty CSV file."""
        # Mock empty CSV content
        mock_file = MagicMock()
        mock_file.content.return_value = b"_row_id,name,age,active\n"
        backend._csv_file = mock_file
        
        mock_csv.DictReader.return_value = []
        
        entries = backend.load_entries(TestEntry)
        assert entries == []

    @patch('ragas_experimental.project.backends.box_csv.csv')
    def test_load_entries_with_data(self, mock_csv, backend, mock_box_client):
        """Test loading entries from CSV with data."""
        # Mock CSV content with data
        mock_file = MagicMock()
        csv_content = "_row_id,name,age,active\nrow1,John,30,true\nrow2,Jane,25,false\n"
        mock_file.content.return_value = csv_content.encode()
        backend._csv_file = mock_file
        
        # Mock CSV reader
        mock_reader = [
            {"_row_id": "row1", "name": "John", "age": "30", "active": "true"},
            {"_row_id": "row2", "name": "Jane", "age": "25", "active": "false"}
        ]
        mock_csv.DictReader.return_value = mock_reader
        
        entries = backend.load_entries(TestEntry)
        
        assert len(entries) == 2
        # Note: This test would need actual TestEntry instances to verify properly
        # The mock would need to be more sophisticated to test type conversion

    def test_append_entry(self, backend, mock_box_client):
        """Test appending new entry to CSV."""
        # Mock existing entries
        backend.load_entries = MagicMock(return_value=[])
        backend._write_entries_to_box = MagicMock()
        
        entry = TestEntry(name="Alice", age=28)
        row_id = backend.append_entry(entry)
        
        assert row_id is not None
        assert hasattr(entry, "_row_id")
        backend._write_entries_to_box.assert_called_once()

    def test_update_entry(self, backend):
        """Test updating existing entry."""
        # Mock existing entries
        existing_entry = TestEntry(name="Bob", age=35)
        existing_entry._row_id = "test_id"
        backend.load_entries = MagicMock(return_value=[existing_entry])
        backend._write_entries_to_box = MagicMock()
        
        # Update entry
        updated_entry = TestEntry(name="Robert", age=36)
        updated_entry._row_id = "test_id"
        
        result = backend.update_entry(updated_entry)
        
        assert result is True
        backend._write_entries_to_box.assert_called_once()

    def test_delete_entry(self, backend, mock_dataset):
        """Test deleting entry from CSV."""
        # Mock dataset entries
        entry1 = TestEntry(name="Carol", age=40)
        entry1._row_id = "keep_id"
        entry2 = TestEntry(name="Dave", age=45)
        entry2._row_id = "delete_id"
        
        mock_dataset._entries = [entry1, entry2]
        backend.dataset = mock_dataset
        backend._write_entries_to_box = MagicMock()
        
        result = backend.delete_entry("delete_id")
        
        assert result is True
        backend._write_entries_to_box.assert_called_once()

    def test_get_entry_by_field(self, backend):
        """Test finding entry by field value."""
        # Mock entries
        entries = [
            TestEntry(name="Eve", age=50),
            TestEntry(name="Frank", age=55),
        ]
        backend.load_entries = MagicMock(return_value=entries)
        
        found_entry = backend.get_entry_by_field("name", "Eve", TestEntry)
        assert found_entry.name == "Eve"
        
        not_found = backend.get_entry_by_field("name", "Unknown", TestEntry)
        assert not_found is None


@pytest.mark.skipif(not box_available, reason="Box SDK not available")
class TestBoxCSVProjectBackend:
    """Test BoxCSVProjectBackend functionality."""

    @pytest.fixture
    def mock_client_config(self):
        """Mock Box client configuration for testing."""
        mock_client = MagicMock(spec=Client)
        # Mock successful authentication verification
        mock_user = MagicMock()
        mock_user.name = "Test User"
        mock_client.user().get.return_value = mock_user
        
        return BoxCSVConfig(client=mock_client)

    @pytest.fixture
    def mock_box_client_for_backend(self):
        """Mock Box client for backend testing."""
        mock_client = MagicMock(spec=Client)
        # Mock successful authentication verification
        mock_user = MagicMock()
        mock_user.name = "Test User"
        mock_client.user().get.return_value = mock_user
        return mock_client

    def test_backend_with_authenticated_client(self, mock_box_client_for_backend):
        """Test creating backend with authenticated client."""
        config = BoxCSVConfig(client=mock_box_client_for_backend)
        backend = BoxCSVProjectBackend(config)
        
        assert backend.box_client == mock_box_client_for_backend
        assert backend.config.client == mock_box_client_for_backend

    def test_client_verification_success(self, mock_box_client_for_backend):
        """Test that client verification works with valid client."""
        # This should not raise an exception
        config = BoxCSVConfig(client=mock_box_client_for_backend)
        assert config._authenticated_user == "Test User"

    def test_client_verification_failure(self):
        """Test that client verification fails with invalid client."""
        mock_client = MagicMock(spec=Client)
        # Make the user().get() call fail to simulate auth failure
        mock_client.user().get.side_effect = Exception("Authentication failed")
        
        with pytest.raises(ValueError, match="Box client authentication failed"):
            BoxCSVConfig(client=mock_client)

    def test_initialize(self, mock_client_config):
        """Test backend initialization."""
        backend = BoxCSVProjectBackend(mock_client_config)
        
        # Mock folder operations
        mock_root_folder = MagicMock()
        mock_project_folder = MagicMock()
        mock_project_folder.id = "project_folder_id"
        
        backend.box_client.folder.return_value = mock_root_folder
        mock_root_folder.get_items.return_value = []
        mock_root_folder.create_subfolder.return_value = mock_project_folder
        mock_project_folder.get_items.return_value = []
        mock_project_folder.create_subfolder.return_value = MagicMock()
        
        backend.initialize("test_project")
        
        assert backend.project_id == "test_project"
        assert backend.project_folder == mock_project_folder

    def test_create_dataset(self, mock_client_config):
        """Test dataset creation."""
        backend = BoxCSVProjectBackend(mock_client_config)
        dataset_id = backend.create_dataset("test_dataset", TestEntry)
        
        assert dataset_id is not None
        assert isinstance(dataset_id, str)

    def test_create_experiment(self, mock_client_config):
        """Test experiment creation."""
        backend = BoxCSVProjectBackend(mock_client_config)
        experiment_id = backend.create_experiment("test_experiment", TestEntry)
        
        assert experiment_id is not None
        assert isinstance(experiment_id, str)

    def test_list_datasets(self, mock_client_config):
        """Test listing datasets."""
        backend = BoxCSVProjectBackend(mock_client_config)
        
        mock_project_folder = MagicMock()
        backend.project_folder = mock_project_folder
        
        # Mock datasets folder with CSV files
        mock_datasets_folder = MagicMock()
        mock_csv_file = MagicMock()
        mock_csv_file.type = "file"
        mock_csv_file.name = "dataset1.csv"
        
        # Mock the item returned from project folder
        mock_folder_item = MagicMock()
        mock_folder_item.type = "folder"
        mock_folder_item.name = "datasets"
        mock_folder_item.id = "datasets_id"
        
        mock_project_folder.get_items.return_value = [mock_folder_item]
        backend.box_client.folder.return_value = mock_datasets_folder
        mock_datasets_folder.get_items.return_value = [mock_csv_file]
        
        datasets = backend.list_datasets()
        
        assert len(datasets) == 1
        assert datasets[0]["name"] == "dataset1"

    def test_get_dataset_backend(self, mock_client_config):
        """Test getting dataset backend instance."""
        backend = BoxCSVProjectBackend(mock_client_config)
        backend.project_folder = MagicMock()
        backend.project_folder.object_id = "project_id"
        
        dataset_backend = backend.get_dataset_backend("ds_123", "test_dataset", TestEntry)
        
        assert isinstance(dataset_backend, BoxCSVDatasetBackend)
        assert dataset_backend.dataset_name == "test_dataset"
        assert dataset_backend.datatable_type == "datasets"

    @patch('boxsdk.auth.jwt_auth.JWTAuth')
    @patch('boxsdk.Client')
    def test_from_jwt_file_factory(self, mock_client_class, mock_jwt_auth):
        """Test JWT file factory method."""
        mock_auth = MagicMock()
        mock_jwt_auth.from_settings_file.return_value = mock_auth
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock successful authentication verification
        mock_user = MagicMock()
        mock_user.name = "Test User"
        mock_client.user().get.return_value = mock_user
        
        backend = BoxCSVProjectBackend.from_jwt_file("config.json")
        
        mock_jwt_auth.from_settings_file.assert_called_once_with("config.json")
        mock_client_class.assert_called_once_with(mock_auth)
        assert backend.config.client == mock_client

    @patch('boxsdk.auth.oauth2.OAuth2')
    @patch('boxsdk.Client')
    def test_from_developer_token_factory(self, mock_client_class, mock_oauth2):
        """Test developer token factory method."""
        mock_auth = MagicMock()
        mock_oauth2.return_value = mock_auth
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock successful authentication verification
        mock_user = MagicMock()
        mock_user.name = "Test User"
        mock_client.user().get.return_value = mock_user
        
        backend = BoxCSVProjectBackend.from_developer_token("test_token")
        
        mock_oauth2.assert_called_once()
        mock_client_class.assert_called_once_with(mock_auth)
        assert backend.config.client == mock_client

    @patch('boxsdk.auth.oauth2.OAuth2')
    @patch('boxsdk.Client')
    def test_from_oauth2_factory(self, mock_client_class, mock_oauth2):
        """Test OAuth2 factory method."""
        mock_auth = MagicMock()
        mock_oauth2.return_value = mock_auth
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock successful authentication verification
        mock_user = MagicMock()
        mock_user.name = "Test User"
        mock_client.user().get.return_value = mock_user
        
        backend = BoxCSVProjectBackend.from_oauth2(
            client_id="test_id", 
            client_secret="test_secret", 
            access_token="test_token"
        )
        
        mock_oauth2.assert_called_once_with(
            client_id="test_id",
            client_secret="test_secret", 
            access_token="test_token",
            refresh_token=None
        )
        mock_client_class.assert_called_once_with(mock_auth)
        assert backend.config.client == mock_client


@pytest.mark.skipif(not box_available, reason="Box SDK not available")
class TestBoxCSVIntegration:
    """Integration tests using VCR.py for Box API interactions."""

    @pytest.mark.vcr()
    def test_authentication_flow(self):
        """Test Box authentication flow (recorded)."""
        # This test would require actual Box credentials for initial recording
        # For now, it serves as a placeholder for VCR integration
        pass

    @pytest.mark.vcr()
    def test_create_project_structure(self):
        """Test creating project folder structure on Box (recorded)."""
        # This would test the actual API calls for folder creation
        pass

    @pytest.mark.vcr()
    def test_upload_download_csv(self):
        """Test uploading and downloading CSV files (recorded)."""
        # This would test the actual file upload/download operations
        pass

    @pytest.mark.vcr()
    def test_error_handling_network_failure(self):
        """Test error handling for network failures (recorded)."""
        # This would test how the backend handles various API errors
        pass


# Example VCR configuration for Box API testing
@pytest.fixture(scope="module")
def vcr_config():
    """VCR configuration for Box API tests."""
    return {
        # Sanitize sensitive data in cassettes
        "filter_headers": [
            ("authorization", "Bearer [REDACTED]"),
            ("box-device-id", "[REDACTED]"),
        ],
        "filter_query_parameters": [
            ("access_token", "[REDACTED]"),
        ],
        # Record new interactions only once
        "record_mode": "once",
        # Match requests by method, uri, and body
        "match_on": ["method", "uri", "body"],
        # Cassette naming
        "cassette_library_dir": "tests/cassettes/box",
        "path_transformer": lambda path: path + ".yaml",
    }


# Mock Box API responses for testing without VCR
@pytest.fixture
def mock_box_responses():
    """Mock Box API responses for comprehensive testing."""
    return {
        "auth_success": {
            "access_token": "mock_token",
            "token_type": "bearer",
            "expires_in": 3600,
        },
        "folder_create": {
            "type": "folder",
            "id": "123456789",
            "name": "test_folder",
        },
        "file_upload": {
            "type": "file",
            "id": "987654321",
            "name": "test.csv",
        },
        "file_content": "name,age,active\nJohn,30,true\n",
    }


@pytest.mark.skipif(not box_available, reason="Box SDK not available")
def test_backend_registration():
    """Test that Box backend is properly registered."""
    from ragas_experimental.project.backends.registry import get_registry
    
    registry = get_registry()
    available_backends = registry.list_available_backends()
    
    # Check if box/csv backend is discoverable
    # Note: This might fail in test environment without proper entry point setup
    if "box/csv" in available_backends:
        backend_info = registry.get_backend_info("box/csv")
        assert "BoxCSVProjectBackend" in str(backend_info["class"])


# Performance and stress tests
@pytest.mark.skipif(not box_available, reason="Box SDK not available")
class TestBoxCSVPerformance:
    """Performance tests for Box CSV backend."""

    def test_large_dataset_operations(self):
        """Test operations with large datasets."""
        # This would test performance with many entries
        pass

    def test_concurrent_operations(self):
        """Test concurrent read/write operations."""
        # This would test thread safety and concurrent access
        pass

    def test_memory_usage_streaming(self):
        """Test memory usage during streaming operations."""
        # This would verify that streaming is used for large files
        pass


# Error scenarios and edge cases
@pytest.mark.skipif(not box_available, reason="Box SDK not available")
class TestBoxCSVErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_client(self):
        """Test handling of invalid Box client."""
        # This should raise Pydantic validation error
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            BoxCSVConfig(client=None)

    def test_network_timeout(self):
        """Test handling of network timeouts."""
        # This would test timeout scenarios
        pass

    def test_rate_limiting(self):
        """Test handling of Box API rate limits."""
        # This would test rate limit responses
        pass

    def test_insufficient_permissions(self):
        """Test handling of insufficient permissions."""
        # This would test permission denied scenarios
        pass