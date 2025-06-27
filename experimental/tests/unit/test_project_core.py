import os
import tempfile
import pytest

from ragas_experimental.project.core import Project


def test_local_backend_creation():
    """Test creating a project with local backend creates proper directory structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        local_project = Project.create(
            name="test_local_project",
            description="A test project using local backend",
            backend="local/csv",
            root_dir=temp_dir
        )
        
        # Assert folder exists
        assert os.path.exists(os.path.join(temp_dir, "test_local_project"))
        assert os.path.exists(os.path.join(temp_dir, "test_local_project", "datasets"))
        assert os.path.exists(os.path.join(temp_dir, "test_local_project", "experiments"))


def test_local_backend_deletion():
    """Test deleting a local backend project removes the directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        local_project = Project.create(
            name="test_local_project",
            description="A test project using local backend",
            backend="local/csv",
            root_dir=temp_dir
        )
        
        project_path = os.path.join(temp_dir, "test_local_project")
        assert os.path.exists(project_path)
        
        local_project.delete()
        assert not os.path.exists(project_path)


def test_project_get_existing():
    """Test getting an existing project."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a project
        local_project = Project.create(
            name="test_local_project",
            description="A test project using local backend",
            backend="local/csv",
            root_dir=temp_dir
        )
        
        # Get the project
        retrieved_project = Project.get(
            name="test_local_project",
            backend="local/csv",
            root_dir=temp_dir
        )
        
        assert retrieved_project.name == "test_local_project"
        # Check backend type by checking if it's a LocalCSVProjectBackend
        from ragas_experimental.project.backends.local_csv import LocalCSVProjectBackend
        assert isinstance(retrieved_project._backend, LocalCSVProjectBackend)


def test_project_get_nonexistent():
    """Test getting a non-existent project raises ValueError."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(ValueError, match="Local project 'nonexistent' does not exist"):
            Project.get(
                name="nonexistent",
                backend="local/csv",
                root_dir=temp_dir
            )


def test_project_paths():
    """Test dataset and experiment path generation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        local_project = Project.create(
            name="test_local_project",
            description="A test project using local backend",
            backend="local/csv",
            root_dir=temp_dir
        )
        
        # Test path generation
        dataset_path = local_project.get_dataset_path("example_dataset")
        experiment_path = local_project.get_experiment_path("example_experiment")
        
        expected_dataset_path = os.path.join(temp_dir, "test_local_project", "datasets", "example_dataset.csv")
        expected_experiment_path = os.path.join(temp_dir, "test_local_project", "experiments", "example_experiment.csv")
        
        assert dataset_path == expected_dataset_path
        assert experiment_path == expected_experiment_path


def test_project_repr():
    """Test project string representation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        local_project = Project.create(
            name="test_local_project",
            description="A test project using local backend",
            backend="local/csv",
            root_dir=temp_dir
        )
        
        assert "test_local_project" in str(local_project)
        assert "local/csv" in str(local_project)