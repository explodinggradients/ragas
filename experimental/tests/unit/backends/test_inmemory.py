"""Comprehensive tests for InMemoryBackend for temporary dataset storage.

This test suite has been optimized to reduce redundancy while maintaining full coverage.
Originally 36 tests, now consolidated to 28 tests with identical functionality coverage.
"""

import pytest
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from ragas_experimental.backends.inmemory import InMemoryBackend
from ragas_experimental.backends import get_registry
from ragas_experimental.dataset import Dataset


# Test BaseModel classes
class SimpleTestModel(BaseModel):
    name: str
    age: int
    score: float
    is_active: bool


class ComplexTestModel(BaseModel):
    id: int
    metadata: Dict[str, Any]
    tags: List[str]
    config: Optional[Dict[str, Any]] = None


# Test fixtures
@pytest.fixture
def backend():
    """Create a fresh InMemoryBackend instance for each test."""
    return InMemoryBackend()


@pytest.fixture
def simple_data():
    """Simple test data with basic types."""
    return [
        {"name": "Alice", "age": 30, "score": 85.5, "is_active": True},
        {"name": "Bob", "age": 25, "score": 92.0, "is_active": False},
        {"name": "Charlie", "age": 35, "score": 78.5, "is_active": True},
    ]


@pytest.fixture
def complex_data():
    """Complex test data with nested structures."""
    return [
        {
            "id": 1,
            "metadata": {"score": 0.85, "tags": ["test", "important"]},
            "tags": ["evaluation", "metrics"],
            "config": {"model": "gpt-4", "temperature": 0.7},
        },
        {
            "id": 2,
            "metadata": {"score": 0.92, "tags": ["production"]},
            "tags": ["benchmark", "validation"],
            "config": {"model": "claude-3", "temperature": 0.5},
        },
    ]


# 1. Basic Functionality Tests
class TestInMemoryBackendBasics:
    """Test basic InMemoryBackend functionality.
    
    Consolidated from 14 to 9 tests by combining similar dataset/experiment operations.
    """

    def test_backend_initialization(self):
        """
        Scenario: Initialize InMemoryBackend
        Given: InMemoryBackend class
        When: I create a new instance
        Then: It should initialize with empty storage for datasets and experiments
        """
        backend = InMemoryBackend()
        assert hasattr(backend, "_datasets")
        assert hasattr(backend, "_experiments")
        assert isinstance(backend._datasets, dict)
        assert isinstance(backend._experiments, dict)
        assert len(backend._datasets) == 0
        assert len(backend._experiments) == 0

    def test_save_and_load_operations(self, backend, simple_data):
        """
        Scenario: Save and load datasets and experiments
        Given: An InMemoryBackend instance and sample data
        When: I save and load both datasets and experiments
        Then: The loaded data should match the saved data exactly
        """
        # Test dataset operations
        backend.save_dataset("test_dataset", simple_data)
        loaded_dataset = backend.load_dataset("test_dataset")
        
        assert loaded_dataset == simple_data
        assert len(loaded_dataset) == 3
        assert loaded_dataset[0]["name"] == "Alice"
        assert loaded_dataset[0]["age"] == 30  # Should preserve int type
        assert loaded_dataset[0]["score"] == 85.5  # Should preserve float type
        assert loaded_dataset[0]["is_active"] is True  # Should preserve bool type

        # Test experiment operations
        backend.save_experiment("test_experiment", simple_data)
        loaded_experiment = backend.load_experiment("test_experiment")
        
        assert loaded_experiment == simple_data
        assert len(loaded_experiment) == 3
        assert loaded_experiment[1]["name"] == "Bob"
        assert loaded_experiment[1]["age"] == 25
        assert loaded_experiment[1]["is_active"] is False

    def test_save_and_load_complex_data(self, backend, complex_data):
        """
        Scenario: Save and load complex nested data
        Given: An InMemoryBackend instance and complex nested data
        When: I save and load the data
        Then: All nested structures should be preserved exactly (unlike CSV backend)
        """
        # Save complex data
        backend.save_dataset("complex_dataset", complex_data)

        # Load complex data
        loaded_data = backend.load_dataset("complex_dataset")

        # Verify exact preservation of nested structures
        assert loaded_data == complex_data
        assert loaded_data[0]["metadata"]["score"] == 0.85  # Nested dict preserved
        assert loaded_data[0]["metadata"]["tags"] == [
            "test",
            "important",
        ]  # Nested list preserved
        assert loaded_data[0]["config"]["temperature"] == 0.7  # Nested dict preserved
        assert isinstance(loaded_data[0]["metadata"], dict)  # Type preserved
        assert isinstance(loaded_data[0]["tags"], list)  # Type preserved

    def test_list_empty_operations(self, backend):
        """
        Scenario: List datasets and experiments when none exist
        Given: A fresh InMemoryBackend instance
        When: I call list_datasets() and list_experiments()
        Then: Both should return empty lists
        """
        datasets = backend.list_datasets()
        experiments = backend.list_experiments()
        
        assert datasets == []
        assert experiments == []
        assert isinstance(datasets, list)
        assert isinstance(experiments, list)

    def test_list_operations_after_saving(self, backend, simple_data):
        """
        Scenario: List datasets and experiments after saving multiple items
        Given: An InMemoryBackend instance with saved datasets and experiments
        When: I call list_datasets() and list_experiments()
        Then: Both should return items in sorted order
        """
        # Save multiple datasets
        backend.save_dataset("ds2", simple_data)
        backend.save_dataset("ds1", simple_data)
        
        # Save multiple experiments
        backend.save_experiment("exp2", simple_data)
        backend.save_experiment("exp1", simple_data)

        # List and verify sorted order
        datasets = backend.list_datasets()
        experiments = backend.list_experiments()
        
        assert datasets == ["ds1", "ds2"]
        assert experiments == ["exp1", "exp2"]
        assert len(datasets) == 2
        assert len(experiments) == 2

    def test_save_empty_operations(self, backend):
        """
        Scenario: Save empty datasets and experiments
        Given: An InMemoryBackend instance and empty data lists
        When: I save datasets and experiments with empty data
        Then: Both should save successfully and load as empty lists
        """
        # Save empty dataset
        backend.save_dataset("empty_dataset", [])
        loaded_dataset = backend.load_dataset("empty_dataset")
        
        assert loaded_dataset == []
        assert len(loaded_dataset) == 0
        assert "empty_dataset" in backend.list_datasets()

        # Save empty experiment
        backend.save_experiment("empty_experiment", [])
        loaded_experiment = backend.load_experiment("empty_experiment")
        
        assert loaded_experiment == []
        assert len(loaded_experiment) == 0
        assert "empty_experiment" in backend.list_experiments()

    def test_overwrite_operations(self, backend, simple_data):
        """
        Scenario: Overwrite existing datasets and experiments
        Given: An InMemoryBackend instance with saved datasets and experiments
        When: I save new data to the same names
        Then: The old data should be replaced with new data
        """
        new_data = [{"name": "New", "age": 40, "score": 90.0, "is_active": True}]
        
        # Test dataset overwrite
        backend.save_dataset("test", simple_data)
        initial_data = backend.load_dataset("test")
        assert len(initial_data) == 3
        
        backend.save_dataset("test", new_data)
        loaded_data = backend.load_dataset("test")
        assert loaded_data == new_data
        assert len(loaded_data) == 1
        assert loaded_data[0]["name"] == "New"
        assert backend.list_datasets() == ["test"]

        # Test experiment overwrite
        backend.save_experiment("test_exp", simple_data)
        initial_data = backend.load_experiment("test_exp")
        assert len(initial_data) == 3
        
        backend.save_experiment("test_exp", new_data)
        loaded_data = backend.load_experiment("test_exp")
        assert loaded_data == new_data
        assert len(loaded_data) == 1
        assert loaded_data[0]["name"] == "New"
        assert "test_exp" in backend.list_experiments()

    def test_datasets_and_experiments_separate_storage(self, backend, simple_data):
        """
        Scenario: Datasets and experiments have separate storage
        Given: An InMemoryBackend instance
        When: I save dataset "name1" and experiment "name1" with different data
        Then: Both should be saved independently and retrievable separately
        """
        # Save dataset with name "name1"
        dataset_data = [{"type": "dataset", "value": 1}]
        backend.save_dataset("name1", dataset_data)

        # Save experiment with same name "name1"
        experiment_data = [{"type": "experiment", "value": 2}]
        backend.save_experiment("name1", experiment_data)

        # Verify both are saved independently
        loaded_dataset = backend.load_dataset("name1")
        loaded_experiment = backend.load_experiment("name1")

        assert loaded_dataset == dataset_data
        assert loaded_experiment == experiment_data
        assert loaded_dataset != loaded_experiment

        # Verify both appear in their respective listings
        assert "name1" in backend.list_datasets()
        assert "name1" in backend.list_experiments()

    def test_data_model_parameter_ignored(self, backend, simple_data):
        """
        Scenario: data_model parameter is accepted but ignored
        Given: An InMemoryBackend instance and a Pydantic model
        When: I save dataset/experiment with data_model parameter
        Then: It should save successfully without validation or modification
        """
        # Save dataset with data_model parameter
        backend.save_dataset("test_dataset", simple_data, data_model=SimpleTestModel)

        # Save experiment with data_model parameter
        backend.save_experiment(
            "test_experiment", simple_data, data_model=SimpleTestModel
        )

        # Verify data was saved as-is (no validation or modification)
        loaded_dataset = backend.load_dataset("test_dataset")
        loaded_experiment = backend.load_experiment("test_experiment")

        assert loaded_dataset == simple_data
        assert loaded_experiment == simple_data
        # Verify data is still dict, not model instances
        assert isinstance(loaded_dataset[0], dict)
        assert isinstance(loaded_experiment[0], dict)


# 2. Error Handling Tests
class TestInMemoryBackendErrorHandling:
    """Test error scenarios and edge cases."""

    def test_load_nonexistent_dataset(self, backend):
        """
        Scenario: Load a dataset that doesn't exist
        Given: An InMemoryBackend instance with no saved datasets
        When: I try to load a dataset named "nonexistent"
        Then: It should raise FileNotFoundError with appropriate message
        """
        with pytest.raises(FileNotFoundError) as exc_info:
            backend.load_dataset("nonexistent")

        assert "Dataset 'nonexistent' not found" in str(exc_info.value)

    def test_load_nonexistent_experiment(self, backend):
        """
        Scenario: Load an experiment that doesn't exist
        Given: An InMemoryBackend instance with no saved experiments
        When: I try to load an experiment named "nonexistent"
        Then: It should raise FileNotFoundError with appropriate message
        """
        with pytest.raises(FileNotFoundError) as exc_info:
            backend.load_experiment("nonexistent")

        assert "Experiment 'nonexistent' not found" in str(exc_info.value)

    def test_none_values_handling(self, backend):
        """
        Scenario: Handle None values in data
        Given: An InMemoryBackend instance and data containing None values
        When: I save and load the data
        Then: None values should be preserved exactly
        """
        data_with_none = [
            {"name": "Alice", "age": 30, "optional_field": None},
            {"name": None, "age": 25, "optional_field": "value"},
            {"name": "Charlie", "age": None, "optional_field": None},
        ]

        # Save and load data
        backend.save_dataset("none_test", data_with_none)
        loaded_data = backend.load_dataset("none_test")

        # Verify None values are preserved exactly
        assert loaded_data == data_with_none
        assert loaded_data[0]["optional_field"] is None
        assert loaded_data[1]["name"] is None
        assert loaded_data[2]["age"] is None
        assert loaded_data[2]["optional_field"] is None

    def test_unicode_and_special_characters(self, backend):
        """
        Scenario: Handle unicode and special characters
        Given: An InMemoryBackend instance and data with unicode/special chars
        When: I save and load the data
        Then: All unicode and special characters should be preserved
        """
        unicode_data = [
            {
                "name": "José María",
                "description": "Testing émojis 🚀 and spëcial chars",
                "chinese": "你好世界",
                "symbols": "!@#$%^&*()_+{}[]|;:,.<>?",
                "emoji": "🎉🔥💯",
            }
        ]

        # Save and load data
        backend.save_dataset("unicode_test", unicode_data)
        loaded_data = backend.load_dataset("unicode_test")

        # Verify all unicode and special characters are preserved
        assert loaded_data == unicode_data
        assert loaded_data[0]["name"] == "José María"
        assert loaded_data[0]["chinese"] == "你好世界"
        assert "🚀" in loaded_data[0]["description"]
        assert loaded_data[0]["emoji"] == "🎉🔥💯"
        assert loaded_data[0]["symbols"] == "!@#$%^&*()_+{}[]|;:,.<>?"

    def test_large_dataset_handling(self, backend):
        """
        Scenario: Handle large datasets in memory
        Given: An InMemoryBackend instance and a large dataset
        When: I save and load the large dataset
        Then: All data should be preserved without truncation
        """
        # Create a large dataset (1000 items)
        large_data = [
            {"id": i, "value": f"item_{i}", "large_text": "A" * 1000}
            for i in range(1000)
        ]

        # Save and load large dataset
        backend.save_dataset("large_test", large_data)
        loaded_data = backend.load_dataset("large_test")

        # Verify all data is preserved
        assert len(loaded_data) == 1000
        assert loaded_data == large_data
        assert loaded_data[0]["id"] == 0
        assert loaded_data[999]["id"] == 999
        assert len(loaded_data[0]["large_text"]) == 1000

    def test_deeply_nested_structures(self, backend):
        """
        Scenario: Handle deeply nested data structures
        Given: An InMemoryBackend instance and deeply nested data
        When: I save and load the nested data
        Then: All nested levels should be preserved exactly
        """
        deeply_nested = [
            {
                "level1": {
                    "level2": {
                        "level3": {
                            "level4": {
                                "level5": {
                                    "value": "deep_value",
                                    "list": [1, 2, {"nested_in_list": True}],
                                }
                            }
                        }
                    }
                }
            }
        ]

        # Save and load deeply nested data
        backend.save_dataset("nested_test", deeply_nested)
        loaded_data = backend.load_dataset("nested_test")

        # Verify all nested levels are preserved
        assert loaded_data == deeply_nested
        assert (
            loaded_data[0]["level1"]["level2"]["level3"]["level4"]["level5"]["value"]
            == "deep_value"
        )
        assert (
            loaded_data[0]["level1"]["level2"]["level3"]["level4"]["level5"]["list"][2][
                "nested_in_list"
            ]
            is True
        )


# 3. Integration Tests
class TestInMemoryBackendIntegration:
    """Test integration with other components.
    
    Consolidated from 8 to 6 tests by combining similar integration scenarios.
    """

    def test_backend_registration(self):
        """
        Scenario: InMemoryBackend is registered in the backend registry
        Given: The backend registry system
        When: I check for "inmemory" backend
        Then: It should be available and return InMemoryBackend class
        """
        registry = get_registry()

        # Check that inmemory backend is registered
        assert "inmemory" in registry

        # Check that it returns the correct class
        backend_class = registry["inmemory"]
        assert backend_class == InMemoryBackend

        # Check that we can create an instance
        backend_instance = backend_class()
        assert isinstance(backend_instance, InMemoryBackend)

    def test_dataset_with_inmemory_backend(self, backend, simple_data):
        """
        Scenario: Create Dataset with InMemoryBackend (string and instance)
        Given: Dataset class and InMemoryBackend string/instance
        When: I create Datasets with both backend formats
        Then: Both should create successfully with InMemoryBackend instances
        """
        # Test with backend string
        dataset_string = Dataset("test_dataset_string", "inmemory", data=simple_data)
        assert isinstance(dataset_string.backend, InMemoryBackend)
        assert dataset_string.name == "test_dataset_string"
        assert len(dataset_string) == 3
        
        dataset_string.save()
        loaded_dataset = Dataset.load("test_dataset_string", dataset_string.backend)
        assert len(loaded_dataset) == 3
        assert loaded_dataset[0]["name"] == "Alice"

        # Test with backend instance
        dataset_instance = Dataset("test_dataset_instance", backend, data=simple_data)
        assert dataset_instance.backend is backend
        assert dataset_instance.name == "test_dataset_instance"
        assert len(dataset_instance) == 3
        
        dataset_instance.save()
        loaded_data = backend.load_dataset("test_dataset_instance")
        assert len(loaded_data) == 3
        assert loaded_data[0]["name"] == "Alice"

    def test_dataset_save_and_load_cycle(self, backend, simple_data):
        """
        Scenario: Complete Dataset save and load cycle with inmemory backend
        Given: A Dataset with inmemory backend and sample data
        When: I save the dataset and then load it
        Then: The loaded dataset should contain the original data
        """
        # Create Dataset with inmemory backend
        dataset = Dataset("test_dataset", backend, data=simple_data)
        assert len(dataset) == 3
        
        # Save the dataset
        dataset.save()
        
        # Load the dataset using the same backend instance
        loaded_dataset = Dataset.load("test_dataset", backend)
        
        # Verify the loaded dataset contains the original data
        assert len(loaded_dataset) == 3
        assert loaded_dataset[0]["name"] == "Alice"
        assert loaded_dataset[1]["name"] == "Bob"
        assert loaded_dataset[2]["name"] == "Charlie"
        
        # Verify the data is identical
        for i in range(3):
            assert loaded_dataset[i] == simple_data[i]

    def test_dataset_train_test_split_uses_inmemory(self, simple_data):
        """
        Scenario: train_test_split creates datasets with inmemory backend
        Given: A Dataset with any backend containing sample data
        When: I call train_test_split()
        Then: The returned train and test datasets should use inmemory backend
        """
        # Create Dataset with any backend (let's use a different backend)
        from ragas_experimental.backends.local_csv import LocalCSVBackend
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_backend = LocalCSVBackend(tmp_dir)
            dataset = Dataset("original_dataset", csv_backend, data=simple_data)
            
            # Call train_test_split
            train_dataset, test_dataset = dataset.train_test_split(test_size=0.4, random_state=42)
            
            # Verify train and test datasets use inmemory backend
            assert isinstance(train_dataset.backend, InMemoryBackend)
            assert isinstance(test_dataset.backend, InMemoryBackend)
            
            # Verify original dataset still uses CSV backend
            assert isinstance(dataset.backend, LocalCSVBackend)
            
            # Verify datasets have the expected sizes
            # With 3 items and test_size=0.4: split_index = int(3 * (1 - 0.4)) = int(1.8) = 1
            # So train gets data[:1] = 1 item, test gets data[1:] = 2 items
            assert len(train_dataset) == 1  # train = 60% of 3 = 1.8 -> 1 (int truncation)
            assert len(test_dataset) == 2   # test = 40% of 3 = 1.2 -> 2 (remaining items)
            
            # Verify total data is preserved
            assert len(train_dataset) + len(test_dataset) == 3

    def test_train_test_split_comprehensive(self, simple_data):
        """
        Scenario: train_test_split preserves original backend and maintains data integrity
        Given: Datasets with different backends
        When: I call train_test_split()
        Then: Original backend is preserved and data integrity is maintained
        """
        # Test with CSV backend - preserves original backend
        from ragas_experimental.backends.local_csv import LocalCSVBackend
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_backend = LocalCSVBackend(tmp_dir)
            original_dataset = Dataset("original_dataset", csv_backend, data=simple_data)
            original_backend_id = id(original_dataset.backend)
            
            train_dataset, test_dataset = original_dataset.train_test_split(test_size=0.3, random_state=42)
            
            # Verify original dataset still uses the same CSV backend instance
            assert isinstance(original_dataset.backend, LocalCSVBackend)
            assert id(original_dataset.backend) == original_backend_id
            assert isinstance(train_dataset.backend, InMemoryBackend)
            assert isinstance(test_dataset.backend, InMemoryBackend)
            
            # Verify original dataset data is unchanged
            assert len(original_dataset) == 3
            names = [original_dataset[i]["name"] for i in range(3)]
            assert "Alice" in names and "Bob" in names and "Charlie" in names
        
        # Test with inmemory backend - data integrity
        dataset = Dataset("test_dataset", "inmemory", data=simple_data)
        train_dataset, test_dataset = dataset.train_test_split(test_size=0.33, random_state=42)
        
        # Verify data integrity
        train_data = [dict(item) for item in train_dataset]
        test_data = [dict(item) for item in test_dataset]
        combined_data = train_data + test_data
        
        assert len(combined_data) == len(simple_data)
        for original_item in simple_data:
            assert original_item in combined_data
        assert len(combined_data) == len(set(str(item) for item in combined_data))
        assert isinstance(train_dataset.backend, InMemoryBackend)
        assert isinstance(test_dataset.backend, InMemoryBackend)

    def test_pydantic_model_validation_with_inmemory(self, backend, simple_data):
        """
        Scenario: Pydantic model validation works with inmemory backend
        Given: A Dataset with inmemory backend and Pydantic model
        When: I save and load data with model validation
        Then: Data should be validated and converted to model instances
        """
        # Create Dataset with inmemory backend and Pydantic model validation
        dataset = Dataset("test_dataset", backend, data_model=SimpleTestModel, data=simple_data)
        
        # Save the dataset
        dataset.save()
        
        # Load the dataset with model validation
        loaded_dataset = Dataset.load("test_dataset", backend, data_model=SimpleTestModel)
        
        # Verify data is loaded and validated
        assert len(loaded_dataset) == 3
        
        # Verify all items are SimpleTestModel instances
        for item in loaded_dataset:
            assert isinstance(item, SimpleTestModel)
            assert hasattr(item, 'name')
            assert hasattr(item, 'age')
            assert hasattr(item, 'score')
            assert hasattr(item, 'is_active')
        
        # Verify data values are correct
        assert loaded_dataset[0].name == "Alice"
        assert loaded_dataset[0].age == 30
        assert loaded_dataset[0].score == 85.5
        assert loaded_dataset[0].is_active is True
        
        assert loaded_dataset[1].name == "Bob"
        assert loaded_dataset[1].age == 25
        assert loaded_dataset[1].score == 92.0
        assert loaded_dataset[1].is_active is False


# 4. Isolation and Concurrency Tests
class TestInMemoryBackendIsolation:
    """Test data isolation and concurrency scenarios."""

    def test_multiple_backend_instances_isolation(self, simple_data):
        """
        Scenario: Multiple backend instances don't share data
        Given: Two separate InMemoryBackend instances
        When: I save data in one instance
        Then: The other instance should not have access to that data
        """
        # Create two separate backend instances
        backend1 = InMemoryBackend()
        backend2 = InMemoryBackend()

        # Save data in backend1
        backend1.save_dataset("test_dataset", simple_data)
        backend1.save_experiment("test_experiment", simple_data)

        # Verify backend2 doesn't have access to the data
        with pytest.raises(FileNotFoundError):
            backend2.load_dataset("test_dataset")

        with pytest.raises(FileNotFoundError):
            backend2.load_experiment("test_experiment")

        # Verify backend2 has empty listings
        assert backend2.list_datasets() == []
        assert backend2.list_experiments() == []

        # Verify backend1 still has the data
        assert backend1.list_datasets() == ["test_dataset"]
        assert backend1.list_experiments() == ["test_experiment"]

    def test_concurrent_save_operations(self, simple_data):
        """
        Scenario: Concurrent save operations don't interfere
        Given: An InMemoryBackend instance and multiple concurrent save operations
        When: I save different datasets concurrently
        Then: All saves should complete successfully without data corruption
        """
        import threading
        import time
        
        backend = InMemoryBackend()
        results = []
        
        def save_dataset(dataset_name, data):
            try:
                backend.save_dataset(dataset_name, data)
                results.append(f"success_{dataset_name}")
            except Exception as e:
                results.append(f"error_{dataset_name}_{str(e)}")
        
        # Create multiple threads to save different datasets concurrently
        threads = []
        for i in range(5):
            data = [{"id": i, "name": f"item_{i}", "value": i * 10}]
            thread = threading.Thread(target=save_dataset, args=(f"dataset_{i}", data))
            threads.append(thread)
        
        # Start all threads simultaneously
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all saves completed successfully
        assert len(results) == 5
        for i in range(5):
            assert f"success_dataset_{i}" in results
        
        # Verify all datasets are saved correctly
        datasets = backend.list_datasets()
        assert len(datasets) == 5
        for i in range(5):
            assert f"dataset_{i}" in datasets
            loaded_data = backend.load_dataset(f"dataset_{i}")
            assert loaded_data[0]["id"] == i
            assert loaded_data[0]["value"] == i * 10

    def test_concurrent_read_operations(self, backend, simple_data):
        """
        Scenario: Concurrent read operations are safe
        Given: An InMemoryBackend instance with saved data
        When: I read the same data from multiple threads concurrently
        Then: All reads should return the same correct data
        """
        import threading
        
        # Save initial data
        backend.save_dataset("shared_dataset", simple_data)
        
        results = []
        
        def read_dataset():
            try:
                data = backend.load_dataset("shared_dataset")
                results.append(data)
            except Exception as e:
                results.append(f"error_{str(e)}")
        
        # Create multiple threads to read the same dataset concurrently
        threads = []
        for i in range(10):
            thread = threading.Thread(target=read_dataset)
            threads.append(thread)
        
        # Start all threads simultaneously
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all reads completed successfully
        assert len(results) == 10
        
        # Verify all reads returned the same correct data
        for result in results:
            assert isinstance(result, list)
            assert len(result) == 3
            assert result == simple_data
            assert result[0]["name"] == "Alice"
            assert result[1]["name"] == "Bob"
            assert result[2]["name"] == "Charlie"

    def test_mixed_concurrent_operations(self, backend, simple_data):
        """
        Scenario: Mixed concurrent read/write operations are safe
        Given: An InMemoryBackend instance
        When: I perform concurrent read and write operations
        Then: Operations should complete safely without data corruption
        """
        import threading
        import time
        
        # Save initial data
        backend.save_dataset("mixed_dataset", simple_data)
        
        results = []
        
        def read_operation():
            try:
                data = backend.load_dataset("mixed_dataset")
                results.append(f"read_success_{len(data)}")
            except Exception as e:
                results.append(f"read_error_{str(e)}")
        
        def write_operation(dataset_name, data):
            try:
                backend.save_dataset(dataset_name, data)
                results.append(f"write_success_{dataset_name}")
            except Exception as e:
                results.append(f"write_error_{dataset_name}_{str(e)}")
        
        # Create mixed read and write threads
        threads = []
        
        # Add read threads
        for i in range(3):
            thread = threading.Thread(target=read_operation)
            threads.append(thread)
        
        # Add write threads
        for i in range(3):
            data = [{"id": i, "name": f"concurrent_item_{i}"}]
            thread = threading.Thread(target=write_operation, args=(f"concurrent_dataset_{i}", data))
            threads.append(thread)
        
        # Start all threads simultaneously
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all operations completed successfully
        assert len(results) == 6
        
        # Check that reads succeeded
        read_results = [r for r in results if r.startswith("read_success")]
        assert len(read_results) == 3
        for result in read_results:
            assert "read_success_3" in result  # Should read 3 items
        
        # Check that writes succeeded
        write_results = [r for r in results if r.startswith("write_success")]
        assert len(write_results) == 3
        
        # Verify all datasets exist
        datasets = backend.list_datasets()
        assert "mixed_dataset" in datasets
        for i in range(3):
            assert f"concurrent_dataset_{i}" in datasets

    def test_memory_cleanup_on_overwrite(self, backend, simple_data):
        """
        Scenario: Memory is properly cleaned up when overwriting data
        Given: An InMemoryBackend instance with saved data
        When: I overwrite the data multiple times
        Then: Memory should not grow indefinitely (old data should be cleaned up)
        """
        import sys
        
        # Save initial data
        backend.save_dataset("cleanup_test", simple_data)
        
        # Get initial memory usage (number of datasets should stay constant)
        initial_dataset_count = len(backend.list_datasets())
        
        # Overwrite the same dataset multiple times with different data
        for i in range(100):
            large_data = [{"id": j, "large_text": "X" * 1000} for j in range(i + 1)]
            backend.save_dataset("cleanup_test", large_data)
            
            # Verify dataset count remains constant (no memory leak)
            current_dataset_count = len(backend.list_datasets())
            assert current_dataset_count == initial_dataset_count
            
            # Verify only the latest data is stored
            loaded_data = backend.load_dataset("cleanup_test")
            assert len(loaded_data) == i + 1
            assert loaded_data[0]["id"] == 0
            if i > 0:
                assert loaded_data[i]["id"] == i
        
        # Verify final state
        final_data = backend.load_dataset("cleanup_test")
        assert len(final_data) == 100
        assert final_data[0]["large_text"] == "X" * 1000
        assert final_data[99]["large_text"] == "X" * 1000
        
        # Verify only one dataset still exists
        assert len(backend.list_datasets()) == 1
        assert "cleanup_test" in backend.list_datasets()


# 5. Performance and Edge Cases
class TestInMemoryBackendPerformance:
    """Test performance characteristics and edge cases."""

    def test_complex_data_structure_preservation(self, backend):
        """
        Scenario: Complex data structures are preserved exactly
        Given: An InMemoryBackend instance and complex nested data with various types
        When: I save and load the data
        Then: All data types and structures should be preserved exactly (int, float, bool, None, dict, list)
        """
        complex_types_data = [
            {
                "int_val": 42,
                "float_val": 3.14159,
                "bool_true": True,
                "bool_false": False,
                "none_val": None,
                "string_val": "hello",
                "dict_val": {"nested": "value", "number": 123},
                "list_val": [1, 2.5, True, None, "mixed"],
                "nested_list": [[1, 2], [3, 4]],
                "list_of_dicts": [{"a": 1}, {"b": 2}],
            }
        ]

        # Save and load complex data
        backend.save_dataset("complex_types", complex_types_data)
        loaded_data = backend.load_dataset("complex_types")

        # Verify exact preservation of all types
        assert loaded_data == complex_types_data
        item = loaded_data[0]

        # Check type preservation
        assert type(item["int_val"]) is int
        assert type(item["float_val"]) is float
        assert type(item["bool_true"]) is bool
        assert type(item["bool_false"]) is bool
        assert item["none_val"] is None
        assert type(item["string_val"]) is str
        assert type(item["dict_val"]) is dict
        assert type(item["list_val"]) is list

        # Check nested structure preservation
        assert item["dict_val"]["nested"] == "value"
        assert item["list_val"][0] == 1
        assert item["list_val"][2] is True
        assert item["nested_list"][0] == [1, 2]
        assert item["list_of_dicts"][0]["a"] == 1

    def test_edge_case_dataset_names(self, backend, simple_data):
        """
        Scenario: Handle edge case dataset names
        Given: An InMemoryBackend instance and edge case names (empty, unicode, special chars)
        When: I save datasets with these names
        Then: Names should be handled correctly and datasets should be retrievable
        """
        # Test edge case dataset names
        edge_case_names = [
            "unicode_name_你好",
            "special-chars_name",
            "name.with.dots",
            "name_with_123_numbers",
            "UPPERCASE_NAME",
            "mixed_Case_Name",
        ]

        # Save datasets with edge case names
        for name in edge_case_names:
            backend.save_dataset(name, simple_data)

        # Verify all names are handled correctly
        saved_names = backend.list_datasets()
        for name in edge_case_names:
            assert name in saved_names

        # Verify data can be retrieved with edge case names
        for name in edge_case_names:
            loaded_data = backend.load_dataset(name)
            assert loaded_data == simple_data
