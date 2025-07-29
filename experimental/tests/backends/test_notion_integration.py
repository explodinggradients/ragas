"""Integration test for Notion backend with Dataset system."""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from pydantic import BaseModel

# Check if notion_client is available
try:
    import notion_client
    NOTION_AVAILABLE = True
except ImportError:
    NOTION_AVAILABLE = False

# Import modules - mocking will be done at the test level when needed
from ragas_experimental.dataset import Dataset
from ragas_experimental.backends import get_registry


class TestDataModel(BaseModel):
    """Test model for integration testing."""
    question: str
    answer: str
    score: float


@pytest.fixture
def mock_notion_client():
    """Fixture to provide a mocked Notion client."""
    with patch('ragas_experimental.backends.notion.NotionClient') as mock_client_class:
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_client_instance.databases.retrieve.return_value = {
            "properties": {
                "Name": {"type": "title"},
                "Type": {"type": "select"},
                "Item_Name": {"type": "rich_text"},
                "Data": {"type": "rich_text"},
                "Created_At": {"type": "date"},
                "Updated_At": {"type": "date"}
            }
        }
        mock_client_instance.databases.query.return_value = {"results": []}
        yield mock_client_instance


@pytest.mark.skipif(not NOTION_AVAILABLE, reason="notion_client not installed")
class TestNotionBackendIntegration:
    """Integration tests for Notion backend with the Dataset system."""

    def test_backend_registration(self):
        """Test that Notion backend is properly registered."""
        registry = get_registry()
        registry.discover_backends()
        
        # Check if notion backend is available
        available_backends = list(registry.keys())
        assert "notion" in available_backends, f"Notion backend not found in {available_backends}"

    def test_backend_creation_via_registry(self, mock_notion_client):
        """Test creating Notion backend via registry."""
        registry = get_registry()
        
        try:
            backend = registry.create_backend(
                "notion",
                token="test_token",
                database_id="test_db_id"
            )
            assert backend is not None
            assert hasattr(backend, 'load_dataset')
            assert hasattr(backend, 'save_dataset')
            print("✅ Backend creation via registry successful")
        except Exception as e:
            pytest.fail(f"Failed to create backend via registry: {e}")

    @patch.dict(os.environ, {"NOTION_TOKEN": "test_token", "NOTION_DATABASE_ID": "test_db"})
    def test_dataset_with_notion_backend(self, mock_notion_client):
        """Test Dataset class with Notion backend."""
        try:
            # Create dataset with Notion backend
            dataset = Dataset(
                name="test_integration",
                backend="notion",
                data_model=TestDataModel
            )
            
            assert dataset is not None
            assert dataset.name == "test_integration"
            print("✅ Dataset creation with Notion backend successful")
            
            # Test adding data
            test_record = TestDataModel(
                question="What is integration testing?",
                answer="Testing components together",
                score=0.95
            )
            
            dataset.append(test_record)
            assert len(dataset) == 1
            print("✅ Data append successful")
            
            # Test save operation
            dataset.save()
            assert mock_notion_client.pages.create.called
            print("✅ Save operation successful")
            
        except Exception as e:
            pytest.fail(f"Dataset integration test failed: {e}")

    def test_backend_error_handling_in_dataset(self):
        """Test error handling when using Notion backend with Dataset."""
        # Test with missing environment variables
        with patch.dict(os.environ, {}, clear=True):
            try:
                dataset = Dataset(
                    name="test_error",
                    backend="notion",
                    data_model=TestDataModel
                )
                pytest.fail("Should have raised ValueError for missing config")
            except ValueError as e:
                assert "token required" in str(e).lower()
                print("✅ Proper error handling for missing configuration")

    def test_backend_info_retrieval(self):
        """Test retrieving backend information."""
        registry = get_registry()
        registry.discover_backends()
        
        try:
            info = registry.get_backend_info("notion")
            assert info["name"] == "notion"
            assert "NotionBackend" in str(info["class"])
            assert "notion" in info["module"]
            print("✅ Backend info retrieval successful")
        except KeyError:
            pytest.fail("Notion backend not found in registry")

    def test_list_all_backends_includes_notion(self):
        """Test that notion appears in the list of all backends."""
        registry = get_registry()
        registry.discover_backends()
        
        all_names = registry.list_all_names()
        assert "notion" in all_names
        print("✅ Notion backend appears in all backends list")

    def test_notion_backend_in_print_available(self, capsys):
        """Test that notion backend appears in print_available_backends output."""
        from ragas_experimental.backends import print_available_backends
        
        print_available_backends()
        captured = capsys.readouterr()
        
        # Should contain notion backend information
        assert "notion" in captured.out.lower()
        print("✅ Notion backend appears in available backends output")


@pytest.mark.skipif(NOTION_AVAILABLE, reason="notion_client is installed")
class TestNotionBackendWithoutDependency:
    """Test behavior when notion_client is not available."""
    
    def test_notion_backend_unavailable_error(self):
        """Test that appropriate error is raised when notion_client is not installed."""
        registry = get_registry()
        registry.discover_backends()
        
        # Notion backend should still be discoverable but raise error on creation
        if "notion" in registry.list_all_names():
            with pytest.raises((ImportError, ValueError, RuntimeError)):
                registry.create_backend("notion", token="test", database_id="test")


if __name__ == "__main__":
    # Run basic integration tests
    test_integration = TestNotionBackendIntegration()
    
    print("🧪 Running Notion Backend Integration Tests")
    print("=" * 50)
    
    try:
        test_integration.test_backend_registration()
        test_integration.test_backend_creation_via_registry()
        test_integration.test_dataset_with_notion_backend()
        test_integration.test_backend_error_handling_in_dataset()
        test_integration.test_backend_info_retrieval()
        test_integration.test_list_all_backends_includes_notion()
        
        print("\n🎉 All integration tests passed!")
        print("✅ Notion backend is properly integrated with Ragas experimental system")
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
