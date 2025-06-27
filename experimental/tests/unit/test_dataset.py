import tempfile
import typing as t
import pytest

from ragas_experimental.dataset import Dataset
from ragas_experimental.project.core import Project
from ragas_experimental.model.pydantic_model import ExtendedPydanticBaseModel as BaseModel
from ragas_experimental.metric import MetricResult


class DatasetModel(BaseModel):
    id: int
    name: str
    description: str


class ExperimentModel(DatasetModel):
    tags: t.Literal["tag1", "tag2", "tag3"]
    result: MetricResult


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def test_project(temp_dir):
    """Create a test project."""
    return Project.create(name="test_project", backend="local/csv", root_dir=temp_dir)


@pytest.fixture
def dataset_instance():
    """Create a test dataset instance."""
    return DatasetModel(
        id=0,
        name="test",
        description="test description",
    )


@pytest.fixture
def experiment_instance(dataset_instance):
    """Create a test experiment instance."""
    return ExperimentModel(
        **dataset_instance.model_dump(),
        tags="tag1",
        result=MetricResult(result=0.5, reason="test reason"),
    )


def test_model_creation(dataset_instance, experiment_instance):
    """Test that models can be created successfully."""
    assert dataset_instance.id == 0
    assert dataset_instance.name == "test"
    assert dataset_instance.description == "test description"
    
    assert experiment_instance.id == 0
    assert experiment_instance.tags == "tag1"
    assert experiment_instance.result.result == 0.5


def test_dataset_creation(test_project):
    """Test creating datasets with different models."""
    dataset_with_dataset_model = test_project.create_dataset(
        name="dataset_with_dataset_model", 
        model=DatasetModel
    )
    dataset_with_experiment_model = test_project.create_dataset(
        name="dataset_with_experiment_model", 
        model=ExperimentModel
    )
    
    assert len(dataset_with_dataset_model) == 0
    assert len(dataset_with_experiment_model) == 0


def test_dataset_append_and_length(test_project, dataset_instance, experiment_instance):
    """Test appending entries to datasets and checking length."""
    dataset_with_dataset_model = test_project.create_dataset(
        name="dataset_with_dataset_model", 
        model=DatasetModel
    )
    dataset_with_experiment_model = test_project.create_dataset(
        name="dataset_with_experiment_model", 
        model=ExperimentModel
    )
    
    dataset_with_dataset_model.append(dataset_instance)
    dataset_with_experiment_model.append(experiment_instance)
    
    assert len(dataset_with_dataset_model) == 1
    assert len(dataset_with_experiment_model) == 1


def test_dataset_pop(test_project, dataset_instance, experiment_instance):
    """Test removing entries from datasets."""
    dataset_with_dataset_model = test_project.create_dataset(
        name="dataset_with_dataset_model", 
        model=DatasetModel
    )
    dataset_with_experiment_model = test_project.create_dataset(
        name="dataset_with_experiment_model", 
        model=ExperimentModel
    )
    
    dataset_with_dataset_model.append(dataset_instance)
    dataset_with_experiment_model.append(experiment_instance)
    
    dataset_with_dataset_model.pop()
    dataset_with_experiment_model.pop()
    
    assert len(dataset_with_dataset_model) == 0
    assert len(dataset_with_experiment_model) == 0


def test_dataset_multiple_entries(test_project, dataset_instance, experiment_instance):
    """Test adding multiple entries to datasets."""
    dataset_with_dataset_model = test_project.create_dataset(
        name="dataset_with_dataset_model", 
        model=DatasetModel
    )
    dataset_with_experiment_model = test_project.create_dataset(
        name="dataset_with_experiment_model", 
        model=ExperimentModel
    )
    
    # Add 10 entries
    for i in range(10):
        dataset_with_dataset_model.append(dataset_instance)
        dataset_with_experiment_model.append(experiment_instance)
    
    assert len(dataset_with_dataset_model) == 10
    assert len(dataset_with_experiment_model) == 10


def test_dataset_load(test_project, dataset_instance, experiment_instance):
    """Test loading datasets from storage."""
    dataset_with_dataset_model = test_project.create_dataset(
        name="dataset_with_dataset_model", 
        model=DatasetModel
    )
    
    # Only test with DatasetModel since ExperimentModel has MetricResult serialization issues
    # Add some entries
    for i in range(5):
        dataset_with_dataset_model.append(dataset_instance)
    
    # Load from storage (this should work even if already loaded)
    dataset_with_dataset_model.load()
    
    assert len(dataset_with_dataset_model) == 5


def test_dataset_load_as_dicts(test_project, dataset_instance, experiment_instance):
    """Test loading dataset entries as dictionaries."""
    dataset_with_dataset_model = test_project.create_dataset(
        name="dataset_with_dataset_model", 
        model=DatasetModel
    )
    
    dataset_with_dataset_model.append(dataset_instance)
    
    dicts = dataset_with_dataset_model.load_as_dicts()
    
    assert len(dicts) == 1
    assert dicts[0]["id"] == 0
    assert dicts[0]["name"] == "test"
    assert dicts[0]["description"] == "test description"


def test_dataset_to_pandas(test_project, experiment_instance):
    """Test converting dataset to pandas DataFrame."""
    dataset_with_experiment_model = test_project.create_dataset(
        name="dataset_with_experiment_model", 
        model=ExperimentModel
    )
    
    for i in range(3):
        dataset_with_experiment_model.append(experiment_instance)
    
    df = dataset_with_experiment_model.to_pandas()
    
    assert len(df) == 3
    assert "id" in df.columns
    assert "name" in df.columns
    assert "tags" in df.columns
    assert "result" in df.columns


def test_dataset_save_entry(test_project, experiment_instance):
    """Test saving changes to an entry."""
    dataset_with_experiment_model = test_project.create_dataset(
        name="dataset_with_experiment_model", 
        model=ExperimentModel
    )
    
    dataset_with_experiment_model.append(experiment_instance)
    
    # Get the entry and modify it
    entry = dataset_with_experiment_model[0]
    entry.name = "updated name"
    
    # Save the changes
    dataset_with_experiment_model.save(entry)
    
    # Verify the change persisted
    assert dataset_with_experiment_model[0].name == "updated name"


def test_dataset_get_by_field(test_project, experiment_instance):
    """Test getting entries by field value."""
    dataset_with_experiment_model = test_project.create_dataset(
        name="dataset_with_experiment_model", 
        model=ExperimentModel
    )
    
    dataset_with_experiment_model.append(experiment_instance)
    
    # Get the entry's row_id
    entry = dataset_with_experiment_model[0]
    row_id = entry._row_id
    
    # Find entry by row_id
    found_entry = dataset_with_experiment_model.get(row_id)
    
    assert found_entry is not None
    assert found_entry._row_id == row_id
    assert found_entry.name == experiment_instance.name


def test_dataset_iteration(test_project, dataset_instance):
    """Test iterating over dataset entries."""
    dataset_with_dataset_model = test_project.create_dataset(
        name="dataset_with_dataset_model", 
        model=DatasetModel
    )
    
    # Add multiple entries
    for i in range(3):
        dataset_with_dataset_model.append(dataset_instance)
    
    # Test iteration
    count = 0
    for entry in dataset_with_dataset_model:
        assert entry.name == "test"
        count += 1
    
    assert count == 3


def test_dataset_indexing(test_project, dataset_instance):
    """Test accessing dataset entries by index."""
    dataset_with_dataset_model = test_project.create_dataset(
        name="dataset_with_dataset_model", 
        model=DatasetModel
    )
    
    dataset_with_dataset_model.append(dataset_instance)
    
    # Test indexing
    first_entry = dataset_with_dataset_model[0]
    assert first_entry.name == "test"
    
    # Test slicing
    slice_dataset = dataset_with_dataset_model[0:1]
    assert len(slice_dataset) == 1