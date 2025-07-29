"""Integration test for Notion backend with Dataset system.

This test module includes improved mock assertions that verify the specific
data structure passed to notion_client.pages.create() calls, rather than
just checking if the method was called.
"""

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
            
            # ========================================================================
            # IMPROVED MOCK ASSERTIONS: Check specific data structure passed to API
            # ========================================================================
            # Instead of just: assert mock_notion_client.pages.create.called
            # We now verify the actual data structure passed to pages.create()
            
            # Verify pages.create was called
            assert mock_notion_client.pages.create.called, "pages.create should have been called"
            
            # Check that pages.create was called with expected data structure
            call_args = mock_notion_client.pages.create.call_args
            assert call_args is not None, "pages.create should have been called with arguments"
            
            # Verify the call structure
            args, kwargs = call_args
            assert "parent" in kwargs, "pages.create should have 'parent' parameter"
            assert "properties" in kwargs, "pages.create should have 'properties' parameter"
            
            # Verify parent structure (should reference a database)
            parent = kwargs["parent"]
            assert "database_id" in parent or "type" in parent, "parent should reference a database"
            
            # Verify properties contain expected fields for Notion database structure
            properties = kwargs["properties"]
            expected_fields = ["Name", "Type", "Item_Name", "Data", "Created_At", "Updated_At"]
            for field in expected_fields:
                assert field in properties, f"Expected field '{field}' not found in properties. Available fields: {list(properties.keys())}"
            
            # Verify the data was properly serialized and contains our test record
            assert "Data" in properties, "Data field should be present in properties"
            # The Data field should contain our test record data (usually JSON-serialized)
            data_value = properties["Data"]
            if isinstance(data_value, dict) and "rich_text" in data_value:
                # Notion rich text format
                data_content = data_value["rich_text"][0]["text"]["content"] if data_value["rich_text"] else ""
            else:
                data_content = str(data_value)
            
            # Verify our test data is in the serialized content
            assert "What is integration testing?" in data_content, f"Test question not found in data: {data_content}"
            
            print("✅ Save operation successful with proper data structure verification")
            
            # Additional verification: check call count for more specific assertions
            assert mock_notion_client.pages.create.call_count == 1, f"Expected exactly 1 call to pages.create, got {mock_notion_client.pages.create.call_count}"
            
        except Exception as e:
            pytest.fail(f"Dataset integration test failed: {e}")

    def test_mock_assertion_examples(self, mock_notion_client):
        """Demonstrate different approaches to specific mock assertions."""
        # This test shows various ways to make mock assertions more specific
        # and informative, rather than just checking if a method was called.
        
        with patch.dict(os.environ, {"NOTION_TOKEN": "test_token", "NOTION_DATABASE_ID": "test_db"}):
            dataset = Dataset(
                name="mock_demo",
                backend="notion", 
                data_model=TestDataModel
            )
            
            test_record = TestDataModel(
                question="Demo question?",
                answer="Demo answer",
                score=0.85
            )
            dataset.append(test_record)
            dataset.save()
            
            # Approach 1: Basic call verification (original style)
            assert mock_notion_client.pages.create.called
            
            # Approach 2: Call count verification
            assert mock_notion_client.pages.create.call_count == 1
            
            # Approach 3: Argument inspection (recommended approach)
            call_args = mock_notion_client.pages.create.call_args
            assert call_args is not None
            args, kwargs = call_args
            
            # Approach 4: Using assert_called_with for exact matching
            # Note: This would require knowing the exact expected values
            # mock_notion_client.pages.create.assert_called_with(...)
            
            # Approach 5: Using assert_called_once for call count + verification
            # This ensures the method was called exactly once
            assert mock_notion_client.pages.create.called
            assert mock_notion_client.pages.create.call_count == 1
            
            # Approach 6: Detailed data structure verification (our improved approach)
            properties = kwargs.get("properties", {})
            assert "Data" in properties, "Properties should contain 'Data' field"
            assert "Name" in properties, "Properties should contain 'Name' field"
            
            # Name is in Notion's title format: {'title': [{'text': {'content': 'value'}}]}
            name_prop = properties.get("Name")
            if isinstance(name_prop, dict) and "title" in name_prop:
                actual_name = name_prop["title"][0]["text"]["content"]
            else:
                actual_name = str(name_prop)
            assert actual_name == "mock_demo", f"Expected Name to be 'mock_demo', got {actual_name}"
            
            print("✅ Mock assertion examples completed successfully")

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
                pytest.fail("Should have raised RuntimeError for missing config")
            except RuntimeError as e:
                # The error gets wrapped in RuntimeError by the Dataset class
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
