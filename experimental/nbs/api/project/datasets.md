---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.1
  kernelspec:
    display_name: .venv
    language: python
    name: python3
---

# Dataset Management

> Methods to create and manage datasets within projects

```python
# | default_exp project.datasets
```

```python
# | hide
from nbdev.showdoc import *
```

```python
# | export
import typing as t
import os
import asyncio

from fastcore.utils import patch
from pydantic import BaseModel

from ragas_experimental.project.core import Project
from ragas_experimental.typing import SUPPORTED_BACKENDS
from ragas_experimental.backends.factory import RagasApiClientFactory
from ragas_experimental.backends.ragas_api_client import RagasApiClient
import ragas_experimental.typing as rt
from ragas_experimental.utils import async_to_sync, create_nano_id
from ragas_experimental.dataset import Dataset
```

# | export
import typing as t
import os
import asyncio
import tempfile
import shutil
import csv
from pathlib import Path

from fastcore.utils import patch
from pydantic import BaseModel

from ragas_experimental.project.core import Project
from ragas_experimental.typing import SUPPORTED_BACKENDS
from ragas_experimental.backends.factory import RagasApiClientFactory
from ragas_experimental.backends.ragas_api_client import RagasApiClient
import ragas_experimental.typing as rt
from ragas_experimental.utils import async_to_sync, create_nano_id
from ragas_experimental.dataset import Dataset

# Helper function for tests
def get_test_directory():
    """Create a test directory that will be cleaned up on process exit.
    
    Returns:
        str: Path to test directory
    """
    # Create a directory in the system temp directory
    test_dir = os.path.join(tempfile.gettempdir(), f"ragas_test_{create_nano_id()}")
    os.makedirs(test_dir, exist_ok=True)
    
    return test_dir

```python
#| export
async def create_dataset_columns(project_id, dataset_id, columns, create_dataset_column_func):
    tasks = []
    for column in columns:
        tasks.append(create_dataset_column_func(
            project_id=project_id,
            dataset_id=dataset_id,
            id=create_nano_id(),
            name=column["name"],
            type=column["type"],
            settings=column["settings"],
        ))
    return await asyncio.gather(*tasks)
```

```python
# | export
def get_dataset_from_ragas_app(
    self: Project, 
    name: str, 
    model: t.Type[BaseModel]
) -> Dataset:
    """Create a dataset in the Ragas App backend."""
    # create the dataset
    sync_version = async_to_sync(self._ragas_api_client.create_dataset)
    dataset_info = sync_version(
        project_id=self.project_id,
        name=name if name is not None else model.__name__,
    )

    # create the columns for the dataset
    column_types = rt.ModelConverter.model_to_columns(model)
    sync_version = async_to_sync(create_dataset_columns)
    sync_version(
        project_id=self.project_id,
        dataset_id=dataset_info["id"],
        columns=column_types,
        create_dataset_column_func=self._ragas_api_client.create_dataset_column,
    )
        
    # Return a new Dataset instance
    return Dataset(
        name=name if name is not None else model.__name__,
        model=model,
        project_id=self.project_id,
        dataset_id=dataset_info["id"],
        ragas_api_client=self._ragas_api_client,
        backend="ragas_app"
    )
```

```python
# | export
def get_dataset_from_local(
    self: Project,
    name: str,
    model: t.Type[BaseModel]
) -> Dataset:
    """Create a dataset in the local filesystem backend.
    
    Args:
        name: Name of the dataset
        model: Pydantic model defining the structure
        
    Returns:
        Dataset: A new dataset configured to use the local backend
    """
    # Use a UUID as the dataset ID
    dataset_id = create_nano_id()
    
    # Return a new Dataset instance with local backend
    return Dataset(
        name=name if name is not None else model.__name__,
        model=model,
        project_id=self.project_id,
        dataset_id=dataset_id,
        backend="local",
        local_root_dir=os.path.dirname(self._root_dir)  # Root dir for all projects
    )
```

```python
# | export
@patch
def create_dataset(
    self: Project, 
    model: t.Type[BaseModel], 
    name: t.Optional[str] = None,
    backend: t.Optional[SUPPORTED_BACKENDS] = None
) -> Dataset:
    """Create a new dataset.

    Args:
        model: Model class defining the dataset structure
        name: Name of the dataset (defaults to model name if not provided)
        backend: The backend to use (defaults to project's backend if not specified)

    Returns:
        Dataset: A new dataset object for managing entries
    """
    # If name is not provided, use the model name
    if name is None:
        name = model.__name__
        
    # If backend is not specified, use the project's backend
    if backend is None:
        backend = self.backend

    # Create dataset using the appropriate backend
    if backend == "local":
        return get_dataset_from_local(self, name, model)
    elif backend == "ragas_app":
        return get_dataset_from_ragas_app(self, name, model)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
```

```python
# | export
@patch
def get_dataset_by_id(
    self: Project, 
    dataset_id: str, 
    model: t.Type[BaseModel],
    backend: t.Optional[SUPPORTED_BACKENDS] = None
) -> Dataset:
    """Get an existing dataset by ID.
    
    Args:
        dataset_id: The ID of the dataset to retrieve
        model: The model class to use for the dataset entries
        backend: The backend to use (defaults to project's backend)
        
    Returns:
        Dataset: The retrieved dataset
    """
    # If backend is not specified, use the project's backend
    if backend is None:
        backend = self.backend
        
    if backend == "ragas_app":
        # Search for database with given ID
        sync_version = async_to_sync(self._ragas_api_client.get_dataset)
        dataset_info = sync_version(
            project_id=self.project_id,
            dataset_id=dataset_id
        )

        # For now, return Dataset without model type
        return Dataset(
            name=dataset_info["name"],
            model=model,
            project_id=self.project_id,
            dataset_id=dataset_id,
            ragas_api_client=self._ragas_api_client,
            backend="ragas_app"
        )
    elif backend == "local":
        # For local backend, this is not a typical operation since we use names
        # We could maintain a mapping of IDs to names, but for now just raise an error
        raise NotImplementedError(
            "get_dataset_by_id is not implemented for local backend. "
            "Use get_dataset with the dataset name instead."
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")
```

```python
# | export
@patch
def get_dataset(
    self: Project, 
    dataset_name: str, 
    model: t.Type[BaseModel],
    backend: t.Optional[SUPPORTED_BACKENDS] = None
) -> Dataset:
    """Get an existing dataset by name.
    
    Args:
        dataset_name: The name of the dataset to retrieve
        model: The model class to use for the dataset entries
        backend: The backend to use (defaults to project's backend if not specified)
        
    Returns:
        Dataset: The retrieved dataset
    """
    # If backend is not specified, use the project's backend
    if backend is None:
        backend = self.backend
        
    if backend == "ragas_app":
        # Search for dataset with given name
        sync_version = async_to_sync(self._ragas_api_client.get_dataset_by_name)
        dataset_info = sync_version(
            project_id=self.project_id,
            dataset_name=dataset_name
        )

        # Return Dataset instance
        return Dataset(
            name=dataset_info["name"],
            model=model,
            project_id=self.project_id,
            dataset_id=dataset_info["id"],
            ragas_api_client=self._ragas_api_client,
            backend="ragas_app"
        )
    elif backend == "local":
        # Check if the dataset file exists
        dataset_path = self.get_dataset_path(dataset_name)
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset '{dataset_name}' does not exist")
            
        # Create dataset instance with a random ID
        dataset_id = create_nano_id()
        
        # Return Dataset instance
        return Dataset(
            name=dataset_name,
            model=model,
            project_id=self.project_id,
            dataset_id=dataset_id,
            backend="local",
            local_root_dir=os.path.dirname(self._root_dir)  # Root dir for all projects
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")
```

```python
# | export
@patch
def list_dataset_names(
    self: Project,
    backend: t.Optional[SUPPORTED_BACKENDS] = None
) -> t.List[str]:
    """List all datasets in the project.
    
    Args:
        backend: The backend to use (defaults to project's backend)
        
    Returns:
        List[str]: Names of all datasets in the project
    """
    # If backend is not specified, use the project's backend
    if backend is None:
        backend = self.backend
        
    if backend == "ragas_app":
        # Get all datasets from API
        sync_version = async_to_sync(self._ragas_api_client.list_datasets)
        datasets = sync_version(project_id=self.project_id)
        return [dataset["name"] for dataset in datasets]
    elif backend == "local":
        # Get all CSV files in the datasets directory
        datasets_dir = os.path.join(self._root_dir, "datasets")
        if not os.path.exists(datasets_dir):
            return []
            
        return [
            os.path.splitext(f)[0] for f in os.listdir(datasets_dir)
            if f.endswith('.csv')
        ]
    else:
        raise ValueError(f"Unsupported backend: {backend}")
```

```python
# Example of using the local backend
import tempfile
import os
from pydantic import BaseModel

# Create a temporary directory for demonstration
with tempfile.TemporaryDirectory() as temp_dir:
    # Create a new project with local backend
    local_project = Project.create(
        name="test_local_project",
        description="A test project using local backend",
        backend="local",
        root_dir=temp_dir
    )
    
    # Define a test model
    class LocalTestModel(BaseModel):
        id: int
        name: str
        description: str
        score: float
    
    # Create a dataset with local backend
    local_dataset = local_project.create_dataset(
        model=LocalTestModel,
        name="test_dataset",
        backend="local"
    )
    
    # Check that the dataset file was created
    dataset_path = local_project.get_dataset_path("test_dataset")
    print(f"Dataset file exists: {os.path.exists(dataset_path)}")
    
    # List datasets
    datasets = local_project.list_dataset_names()
    print(f"Datasets in project: {datasets}")
    
    # Get the dataset
    retrieved_dataset = local_project.get_dataset(
        dataset_name="test_dataset",
        model=LocalTestModel,
        backend="local"
    )
    
    print(f"Retrieved dataset: {retrieved_dataset}")
```

```python
# Define a test model for demonstration
class TestModel(BaseModel):
    id: int
    name: str
    description: str
    tags: t.Literal["tag1", "tag2", "tag3"]
    tags_color_coded: t.Annotated[t.Literal["red", "green", "blue"], rt.Select(colors=["red", "green", "blue"])]
    url: t.Annotated[str, rt.Url()] = "https://www.google.com"
```

```python
# Example of using the local backend with Project integration
import tempfile
import os
from pydantic import BaseModel

# Create a temporary directory for demonstration
with tempfile.TemporaryDirectory() as temp_dir:
    # Create a new project with local backend
    local_project = Project.create(
        name="test_local_project",
        description="A test project using local backend",
        backend="local",
        root_dir=temp_dir
    )
    
    # Define a test model
    class LocalTestModel(BaseModel):
        id: int
        name: str
        description: str
        score: float
    
    # Create a dataset with local backend
    local_dataset = local_project.create_dataset(
        model=LocalTestModel,
        name="test_dataset"
    )
    
    # Add some entries
    for i in range(3):
        entry = LocalTestModel(
            id=i,
            name=f"Test Item {i}",
            description=f"Description for item {i}",
            score=i * 0.5
        )
        local_dataset.append(entry)
    
    # Check the dataset
    print(f"Dataset after adding entries: {local_dataset}")
    
    # Get the dataset path
    dataset_path = local_project.get_dataset_path("test_dataset")
    print(f"Dataset file path: {dataset_path}")
    
    # Check that the file exists
    print(f"Dataset file exists: {os.path.exists(dataset_path)}")
    
    # Read CSV content
    with open(dataset_path, 'r') as f:
        csv_content = f.read()
    print(f"CSV content:\n{csv_content}")
    
    # List datasets in the project
    dataset_names = local_project.list_dataset_names()
    print(f"Datasets in project: {dataset_names}")
    
    # Get the dataset by name
    retrieved_dataset = local_project.get_dataset(
        dataset_name="test_dataset",
        model=LocalTestModel
    )
    
    # Load entries
    retrieved_dataset.load()
    print(f"Retrieved dataset: {retrieved_dataset}")
    
    # Modify an entry
    entry = retrieved_dataset[1]
    entry.name = "Updated Name"
    entry.score = 9.9
    retrieved_dataset.save(entry)
    
    # Load again to verify changes
    retrieved_dataset.load()
    print(f"Updated entry: {retrieved_dataset[1]}")
    
    # Convert to DataFrame
    df = retrieved_dataset.to_pandas()
    print("\nDataFrame:")
    print(df)
```

```python
# Example of using ragas_app backend (commented out since it requires API access)
'''
import os
from pydantic import BaseModel

# Set environment variables for API access
RAGAS_APP_TOKEN = "your-api-key"
RAGAS_API_BASE_URL = "https://api.dev.app.ragas.io"
os.environ["RAGAS_APP_TOKEN"] = RAGAS_APP_TOKEN
os.environ["RAGAS_API_BASE_URL"] = RAGAS_API_BASE_URL

# Get a project from the Ragas API
ragas_app_project = Project.get(
    name="Your Project Name",
    backend="ragas_app"
)

# Define a test model
class ApiTestModel(BaseModel):
    id: int
    name: str
    description: str
    score: float

# Create a dataset with ragas_app backend
api_dataset = ragas_app_project.create_dataset(
    model=ApiTestModel,
    name="api_test_dataset",
    backend="ragas_app"
)

# Add some entries
for i in range(3):
    entry = ApiTestModel(
        id=i,
        name=f"API Test Item {i}",
        description=f"Description for API item {i}",
        score=i * 1.1
    )
    api_dataset.append(entry)

# List all datasets in the project
dataset_names = ragas_app_project.list_dataset_names(backend="ragas_app")
print(f"Datasets in project: {dataset_names}")

# Get the dataset by name
retrieved_dataset = ragas_app_project.get_dataset(
    dataset_name="api_test_dataset",
    model=ApiTestModel,
    backend="ragas_app"
)

# Load entries
retrieved_dataset.load()
print(f"Retrieved dataset: {retrieved_dataset}")

# View as DataFrame
df = retrieved_dataset.to_pandas()
print("\nDataFrame:")
print(df)
'''
```

```python
# | export
def update_dataset_class_for_local_backend():
    """Updates the Dataset class to support local backend.
    
    This is called when the module is imported to patch the Dataset class
    with methods that enable local backend support.
    """
    from ragas_experimental.dataset import Dataset
    import csv
    import os
    import uuid
    
    # Add backend parameter to Dataset.__init__
    original_init = Dataset.__init__
    
    def new_init(
        self,
        name: str,
        model: t.Type[BaseModel],
        project_id: str,
        dataset_id: str,
        ragas_api_client=None,
        backend: t.Literal["ragas_app", "local"] = "ragas_app",
        local_root_dir: t.Optional[str] = None,
    ):
        self.backend = backend
        self.local_root_dir = local_root_dir
        
        if backend == "local":
            if local_root_dir is None:
                raise ValueError("local_root_dir is required for local backend")
                
            # Set basic properties
            self.name = name
            self.model = model
            self.project_id = project_id
            self.dataset_id = dataset_id
            self._ragas_api_client = None
            self._entries = []
            
            # Setup column mapping
            if not hasattr(self.model, "__column_mapping__"):
                self.model.__column_mapping__ = {}
                
            # For local backend, columns map directly to field names
            for field_name in model.__annotations__:
                self.model.__column_mapping__[field_name] = field_name
                
            # Load entries from CSV if it exists
            self._load_from_csv()
        else:
            # Call original init for ragas_app backend
            original_init(self, name, model, project_id, dataset_id, ragas_api_client)
    
    # Add method to load from CSV
    def _load_from_csv(self):
        """Load dataset entries from CSV file."""
        if self.backend != "local":
            return
            
        # Construct CSV path
        project_dir = os.path.join(self.local_root_dir, self.project_id)
        csv_path = os.path.join(project_dir, "datasets", f"{self.name}.csv")
        
        if not os.path.exists(csv_path):
            return
            
        # Read CSV
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            
            # Clear existing entries
            self._entries.clear()
            
            # Process rows
            for row in reader:
                try:
                    # Convert types as needed based on model annotations
                    typed_row = {}
                    for field, value in row.items():
                        if field in self.model.__annotations__:
                            field_type = self.model.__annotations__[field]
                            
                            # Handle basic type conversions
                            if field_type == int:
                                typed_row[field] = int(value) if value else 0
                            elif field_type == float:
                                typed_row[field] = float(value) if value else 0.0
                            elif field_type == bool:
                                typed_row[field] = value.lower() in ('true', 't', 'yes', 'y', '1')
                            else:
                                typed_row[field] = value
                    
                    # Create model instance
                    entry = self.model(**typed_row)
                    
                    # Add row_id for tracking changes
                    entry._row_id = str(uuid.uuid4())
                    
                    self._entries.append(entry)
                except Exception as e:
                    print(f"Error loading row: {e}")
    
    # Add method to save to CSV
    def _save_to_csv(self):
        """Save all entries to CSV file."""
        if self.backend != "local":
            return
            
        # Construct CSV path
        project_dir = os.path.join(self.local_root_dir, self.project_id)
        csv_path = os.path.join(project_dir, "datasets", f"{self.name}.csv")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Get field names from model
        field_names = list(self.model.__annotations__.keys())
        
        # Write to CSV
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
            
            for entry in self._entries:
                # Convert model instance to dict and write row
                writer.writerow(entry.model_dump())
    
    # Patch the original methods to support local backend
    
    # Patch append
    original_append = Dataset.append
    
    def new_append(self, entry):
        if self.backend == "local":
            if not isinstance(entry, self.model):
                raise TypeError(f"Entry must be an instance of {self.model.__name__}")
                
            # Add row_id for tracking changes
            entry._row_id = str(uuid.uuid4())
            
            # Add to in-memory entries
            self._entries.append(entry)
            
            # Save to CSV
            self._save_to_csv()
        else:
            original_append(self, entry)
    
    # Patch pop
    original_pop = Dataset.pop
    
    def new_pop(self, index=-1):
        if self.backend == "local":
            # Remove from in-memory entries
            entry = self._entries.pop(index)
            
            # Save to CSV
            self._save_to_csv()
            
            return entry
        else:
            return original_pop(self, index)
    
    # Patch load
    original_load = Dataset.load
    
    def new_load(self):
        if self.backend == "local":
            self._load_from_csv()
        else:
            original_load(self)
    
    # Patch save
    original_save = Dataset.save
    
    def new_save(self, item):
        if self.backend == "local":
            if not isinstance(item, self.model):
                raise TypeError(f"Item must be an instance of {self.model.__name__}")
                
            # Find the item in our entries
            found = False
            for i, entry in enumerate(self._entries):
                if hasattr(entry, "_row_id") and hasattr(item, "_row_id") and entry._row_id == item._row_id:
                    # Update the entry
                    self._entries[i] = item
                    found = True
                    break
                    
            if not found:
                # If we didn't find it, add it
                if not hasattr(item, "_row_id"):
                    item._row_id = str(uuid.uuid4())
                self._entries.append(item)
                
            # Save to CSV
            self._save_to_csv()
        else:
            original_save(self, item)
    
    # Apply all patches
    Dataset.__init__ = new_init
    Dataset._load_from_csv = _load_from_csv
    Dataset._save_to_csv = _save_to_csv
    Dataset.append = new_append
    Dataset.pop = new_pop
    Dataset.load = new_load
    Dataset.save = new_save
    
    return Dataset

# Update the Dataset class
updated_dataset_class = update_dataset_class_for_local_backend()
```

```python
# Example of using the local backend Dataset operations
import tempfile
import os
from pydantic import BaseModel

# Create a temporary directory for demonstration
with tempfile.TemporaryDirectory() as temp_dir:
    # Create a new project with local backend
    local_project = Project.create(
        name="test_local_project",
        description="A test project using local backend",
        backend="local",
        root_dir=temp_dir
    )
    
    # Define a test model
    class LocalTestModel(BaseModel):
        id: int
        name: str
        description: str
        score: float
    
    # Create a dataset with local backend
    local_dataset = local_project.create_dataset(
        model=LocalTestModel,
        name="test_dataset",
        backend="local"
    )
    
    # Add some entries to the dataset
    for i in range(5):
        entry = LocalTestModel(
            id=i,
            name=f"Test Item {i}",
            description=f"Description for item {i}",
            score=i * 0.1
        )
        local_dataset.append(entry)
    
    # Print the dataset contents
    print(f"Dataset after adding entries: {local_dataset}")
    
    # Check the CSV file
    dataset_path = local_project.get_dataset_path("test_dataset")
    print(f"Dataset file path: {dataset_path}")
    with open(dataset_path, 'r') as f:
        csv_content = f.read()
    print(f"CSV content:\n{csv_content}")
    
    # Modify an entry
    entry = local_dataset[2]
    entry.name = "Updated Name"
    entry.score = 9.9
    local_dataset.save(entry)
    
    # Load the dataset again
    local_dataset.load()
    
    # Print updated entry
    print(f"Updated entry: {local_dataset[2]}")
    
    # Convert to pandas DataFrame
    df = local_dataset.to_pandas()
    print("\nDataFrame:")
    print(df)
```
