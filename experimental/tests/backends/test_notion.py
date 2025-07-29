"""Tests for the Notion backend implementation."""

import json
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from pydantic import BaseModel

# Check if notion_client is available
try:
    import notion_client
    NOTION_INSTALLED = True
except ImportError:
    NOTION_INSTALLED = False

# Import the backend - tests will handle mocking as needed
from ragas_experimental.backends.notion import NotionBackend


class TestRecord(BaseModel):
    """Test data model for testing."""
    question: str
    answer: str
    score: float


@pytest.mark.skipif(not NOTION_INSTALLED, reason="notion_client not installed")
class TestNotionBackend:
    """Test suite for NotionBackend."""

    @pytest.fixture
    def mock_notion_available(self):
        """Mock NOTION_AVAILABLE to be True for tests."""
        with patch('ragas_experimental.backends.notion.NOTION_AVAILABLE', True):
            yield

    @pytest.fixture
    def mock_notion_client(self):
        """Create a mock Notion client."""
        client = Mock()
        return client

    @pytest.fixture
    def mock_database_response(self):
        """Mock database structure response."""
        return {
            "properties": {
                "Name": {"type": "title"},
                "Type": {"type": "select", "select": {"options": [
                    {"name": "dataset"}, {"name": "experiment"}
                ]}},
                "Item_Name": {"type": "rich_text"},
                "Data": {"type": "rich_text"},
                "Created_At": {"type": "date"},
                "Updated_At": {"type": "date"}
            }
        }

    @pytest.fixture
    def sample_data(self):
        """Sample test data."""
        return [
            {"question": "What is AI?", "answer": "Artificial Intelligence", "score": 0.95},
            {"question": "What is ML?", "answer": "Machine Learning", "score": 0.88}
        ]

    @pytest.fixture
    def mock_query_response(self):
        """Mock query response from Notion."""
        return {
            "results": [
                {
                    "id": "page1",
                    "properties": {
                        "Name": {"title": [{"plain_text": "test_dataset"}]},
                        "Type": {"select": {"name": "dataset"}},
                        "Item_Name": {"rich_text": [{"plain_text": "item_0"}]},
                        "Data": {"rich_text": [{"plain_text": '{"question": "What is AI?", "answer": "Artificial Intelligence", "score": 0.95}'}]},
                        "Created_At": {"date": {"start": "2024-01-01T00:00:00"}},
                        "Updated_At": {"date": {"start": "2024-01-01T00:00:00"}}
                    }
                }
            ]
        }

    def test_notion_availability(self, mock_notion_available):
        """Test that we can check notion availability."""
        # During testing, we mock NOTION_AVAILABLE to be True
        from ragas_experimental.backends.notion import NOTION_AVAILABLE
        assert NOTION_AVAILABLE is True

    @patch.dict(os.environ, {"NOTION_TOKEN": "test_token", "NOTION_DATABASE_ID": "test_db_id"})
    def test_init_with_env_vars(self, mock_notion_available, mock_notion_client, mock_database_response):
        """Test initialization with environment variables."""
        with patch('ragas_experimental.backends.notion.NotionClient') as mock_client_class:
            mock_client_instance = Mock()
            mock_client_class.return_value = mock_client_instance
            mock_client_instance.databases.retrieve.return_value = mock_database_response
            
            backend = NotionBackend()
            
            assert backend.token == "test_token"
            assert backend.database_id == "test_db_id"
            mock_client_class.assert_called_once_with(auth="test_token")

    def test_init_with_explicit_params(self, mock_notion_available, mock_database_response):
        """Test initialization with explicit parameters."""
        with patch('ragas_experimental.backends.notion.NotionClient') as mock_client_class:
            mock_client_instance = Mock()
            mock_client_class.return_value = mock_client_instance
            mock_client_instance.databases.retrieve.return_value = mock_database_response
            
            backend = NotionBackend(token="explicit_token", database_id="explicit_db")
            
            assert backend.token == "explicit_token"
            assert backend.database_id == "explicit_db"

    def test_init_missing_token(self, mock_notion_available):
        """Test initialization fails with missing token."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Notion token required"):
                NotionBackend(database_id="test_db")

    def test_init_missing_database_id(self, mock_notion_available):
        """Test initialization fails with missing database ID."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Notion database ID required"):
                NotionBackend(token="test_token")

    def test_init_invalid_database(self, mock_notion_available):
        """Test initialization fails with invalid database."""
        with patch('ragas_experimental.backends.notion.NotionClient') as mock_client_class:
            mock_client_instance = Mock()
            mock_client_class.return_value = mock_client_instance
            
            # Mock API error - use generic Exception since APIResponseError may not be available
            mock_client_instance.databases.retrieve.side_effect = Exception("Database not found")
            
            with pytest.raises(ValueError, match="Cannot access Notion database"):
                NotionBackend(token="test_token", database_id="invalid_db")

    def test_database_structure_validation(self, mock_notion_available):
        """Test database structure validation."""
        with patch('ragas_experimental.backends.notion.NotionClient') as mock_client_class:
            mock_client_instance = Mock()
            mock_client_class.return_value = mock_client_instance
            
            # Missing required properties
            mock_client_instance.databases.retrieve.return_value = {
                "properties": {
                    "Name": {"type": "title"}
                    # Missing other required properties
                }
            }
            
            with pytest.raises(ValueError, match="Database missing required properties"):
                NotionBackend(token="test_token", database_id="test_db")

    def test_convert_to_notion_properties(self, mock_notion_available, mock_database_response):
        """Test data conversion to Notion properties."""
        with patch('ragas_experimental.backends.notion.NotionClient') as mock_client_class:
            mock_client_instance = Mock()
            mock_client_class.return_value = mock_client_instance
            mock_client_instance.databases.retrieve.return_value = mock_database_response
            
            backend = NotionBackend(token="test_token", database_id="test_db")
            
            data = {
                "Name": "test_dataset",
                "Type": "dataset",
                "Item_Name": "item_1",
                "Data": '{"question": "test"}',
                "Created_At": "2024-01-01T00:00:00"
            }
            
            result = backend._convert_to_notion_properties(data)
            
            assert result["Name"]["title"][0]["text"]["content"] == "test_dataset"
            assert result["Type"]["select"]["name"] == "dataset"
            assert result["Item_Name"]["rich_text"][0]["text"]["content"] == "item_1"

    def test_convert_from_notion_properties(self, mock_notion_available, mock_database_response):
        """Test data conversion from Notion properties."""
        with patch('ragas_experimental.backends.notion.NotionClient') as mock_client_class:
            mock_client_instance = Mock()
            mock_client_class.return_value = mock_client_instance
            mock_client_instance.databases.retrieve.return_value = mock_database_response
            
            backend = NotionBackend(token="test_token", database_id="test_db")
            
            page = {
                "properties": {
                    "Name": {"title": [{"plain_text": "test_dataset"}]},
                    "Type": {"select": {"name": "dataset"}},
                    "Data": {"rich_text": [{"plain_text": '{"question": "test", "score": 0.9}'}]}
                }
            }
            
            result = backend._convert_from_notion_properties(page)
            
            assert result["Name"] == "test_dataset"
            assert result["Type"] == "dataset"
            assert isinstance(result["Data"], dict)
            assert result["Data"]["question"] == "test"

    def test_query_database(self, mock_notion_available, mock_database_response, mock_query_response):
        """Test database querying."""
        with patch('ragas_experimental.backends.notion.NotionClient') as mock_client_class:
            mock_client_instance = Mock()
            mock_client_class.return_value = mock_client_instance
            mock_client_instance.databases.retrieve.return_value = mock_database_response
            mock_client_instance.databases.query.return_value = mock_query_response
            
            backend = NotionBackend(token="test_token", database_id="test_db")
            
            # Test query with type filter only
            result = backend._query_database("datasets")
            assert len(result) == 1
            
            # Test query with type and name filter
            result = backend._query_database("datasets", "test_dataset")
            assert len(result) == 1

    def test_save_dataset(self, mock_notion_available, mock_database_response, sample_data):
        """Test saving a dataset."""
        with patch('ragas_experimental.backends.notion.NotionClient') as mock_client_class:
            mock_client_instance = Mock()
            mock_client_class.return_value = mock_client_instance
            mock_client_instance.databases.retrieve.return_value = mock_database_response
            
            # Mock existing pages (to be archived)
            mock_client_instance.databases.query.return_value = {"results": []}
            
            backend = NotionBackend(token="test_token", database_id="test_db")
            backend.save_dataset("test_dataset", sample_data)
            
            # Should create one page per data item
            assert mock_client_instance.pages.create.call_count == len(sample_data)

    def test_load_dataset(self, mock_notion_available, mock_database_response, mock_query_response):
        """Test loading a dataset."""
        with patch('ragas_experimental.backends.notion.NotionClient') as mock_client_class:
            mock_client_instance = Mock()
            mock_client_class.return_value = mock_client_instance
            mock_client_instance.databases.retrieve.return_value = mock_database_response
            mock_client_instance.databases.query.return_value = mock_query_response
            
            backend = NotionBackend(token="test_token", database_id="test_db")
            result = backend.load_dataset("test_dataset")
            
            assert len(result) == 1
            assert result[0]["question"] == "What is AI?"

    def test_load_nonexistent_dataset(self, mock_notion_available, mock_database_response):
        """Test loading a nonexistent dataset raises FileNotFoundError."""
        with patch('ragas_experimental.backends.notion.NotionClient') as mock_client_class:
            mock_client_instance = Mock()
            mock_client_class.return_value = mock_client_instance
            mock_client_instance.databases.retrieve.return_value = mock_database_response
            mock_client_instance.databases.query.return_value = {"results": []}
            
            backend = NotionBackend(token="test_token", database_id="test_db")
            
            with pytest.raises(FileNotFoundError):
                backend.load_dataset("nonexistent")

    def test_list_datasets(self, mock_notion_available, mock_database_response):
        """Test listing datasets."""
        with patch('ragas_experimental.backends.notion.NotionClient') as mock_client_class:
            mock_client_instance = Mock()
            mock_client_class.return_value = mock_client_instance
            mock_client_instance.databases.retrieve.return_value = mock_database_response
            
            # Mock response with multiple datasets
            mock_client_instance.databases.query.return_value = {
                "results": [
                    {
                        "properties": {
                            "Name": {"title": [{"plain_text": "dataset_1"}]},
                            "Type": {"select": {"name": "dataset"}}
                        }
                    },
                    {
                        "properties": {
                            "Name": {"title": [{"plain_text": "dataset_2"}]},
                            "Type": {"select": {"name": "dataset"}}
                        }
                    }
                ]
            }
            
            backend = NotionBackend(token="test_token", database_id="test_db")
            datasets = backend.list_datasets()
            
            assert "dataset_1" in datasets
            assert "dataset_2" in datasets
            assert len(datasets) == 2

    def test_experiment_operations(self, mock_notion_available, mock_database_response, sample_data):
        """Test experiment save/load operations."""
        with patch('ragas_experimental.backends.notion.NotionClient') as mock_client_class:
            mock_client_instance = Mock()
            mock_client_class.return_value = mock_client_instance
            mock_client_instance.databases.retrieve.return_value = mock_database_response
            mock_client_instance.databases.query.return_value = {"results": []}
            
            backend = NotionBackend(token="test_token", database_id="test_db")
            
            # Test save experiment
            backend.save_experiment("test_experiment", sample_data)
            assert mock_client_instance.pages.create.call_count == len(sample_data)
            
            # Test list experiments
            mock_client_instance.databases.query.return_value = {
                "results": [
                    {
                        "properties": {
                            "Name": {"title": [{"plain_text": "exp_1"}]},
                            "Type": {"select": {"name": "experiment"}}
                        }
                    }
                ]
            }
            
            experiments = backend.list_experiments()
            assert "exp_1" in experiments

    def test_repr(self, mock_notion_available, mock_database_response):
        """Test string representation."""
        with patch('ragas_experimental.backends.notion.NotionClient') as mock_client_class:
            mock_client_instance = Mock()
            mock_client_class.return_value = mock_client_instance
            mock_client_instance.databases.retrieve.return_value = mock_database_response
            
            backend = NotionBackend(token="test_token", database_id="test_database_id")
            repr_str = repr(backend)
            
            assert "NotionBackend" in repr_str
            assert "test_dat" in repr_str  # Truncated database ID (first 8 chars)

    def test_data_truncation(self, mock_notion_available, mock_database_response):
        """Test that large data gets truncated."""
        with patch('ragas_experimental.backends.notion.NotionClient') as mock_client_class:
            mock_client_instance = Mock()
            mock_client_class.return_value = mock_client_instance
            mock_client_instance.databases.retrieve.return_value = mock_database_response
            
            backend = NotionBackend(token="test_token", database_id="test_db")
            
            # Create data that's too long
            large_data = {"Data": "x" * 3000}  # Exceeds 2000 char limit
            
            result = backend._convert_to_notion_properties(large_data)
            content = result["Data"]["rich_text"][0]["text"]["content"]
            
            assert len(content) <= 2000
            assert content.endswith("...")

    @patch.dict(os.environ, {"NOTION_TOKEN": "test_token", "NOTION_DATABASE_ID": "test_db"})
    def test_import_error_handling(self, mock_notion_available):
        """Test proper handling when notion_client is not available."""
        # Test when NOTION_AVAILABLE is False
        with patch('ragas_experimental.backends.notion.NOTION_AVAILABLE', False):
            with pytest.raises(ImportError, match="Notion backend requires additional dependencies"):
                NotionBackend()


@pytest.mark.skipif(NOTION_INSTALLED, reason="notion_client is installed")
class TestNotionBackendWithoutDependency:
    """Test behavior when notion_client is not available."""
    
    def test_notion_backend_unavailable_error(self):
        """Test that appropriate error is raised when notion_client is not installed."""
        # This test will run when notion_client is not installed
        # The actual NOTION_AVAILABLE should be False in this environment
        with pytest.raises(ImportError, match="Notion backend requires additional dependencies"):
            NotionBackend()


class TestNotionBackendIntegration:
    """Integration tests for NotionBackend (these would require real API)."""

    @pytest.mark.skip("Requires real Notion API credentials")
    def test_real_api_integration(self):
        """Test with real Notion API (requires manual setup)."""
        # This test would require:
        # 1. Real NOTION_TOKEN and NOTION_DATABASE_ID environment variables
        # 2. A test database with proper structure
        # 3. Network access to Notion API
        
        backend = NotionBackend()
        
        # Test basic operations
        test_data = [{"test": "data", "score": 0.5}]
        backend.save_dataset("integration_test", test_data)
        
        loaded_data = backend.load_dataset("integration_test")
        assert len(loaded_data) == 1
        
        datasets = backend.list_datasets()
        assert "integration_test" in datasets


# Test fixtures for pytest
@pytest.fixture(scope="session")
def notion_backend_test_setup():
    """Set up test environment for Notion backend tests."""
    # Mock the notion_client module for all tests
    with patch.dict('sys.modules', {
        'notion_client': MagicMock(),
        'notion_client.errors': MagicMock()
    }):
        yield
