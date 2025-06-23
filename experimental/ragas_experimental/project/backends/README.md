# Backend Development Guide

This guide shows you how to add new storage backends to the Ragas project system. The backend architecture supports multiple storage solutions like CSV files, databases, cloud platforms, and more.

## Architecture Overview

The backend system uses a two-layer architecture:

1. **ProjectBackend**: Manages project-level operations (creating datasets/experiments, listing, etc.)
2. **DatasetBackend**: Handles individual dataset operations (reading/writing entries, CRUD operations)

```python
# High-level flow
Project -> ProjectBackend -> DatasetBackend -> Storage (CSV, DB, API, etc.)
```

### Plugin System

Backends can be added in two ways:
- **Internal backends**: Built into the main codebase
- **External plugins**: Distributed as separate pip packages

The system uses a registry pattern with automatic discovery via setuptools entry points.

## Section 1: Adding Internal Backends

Follow these steps to add a new backend to the main ragas_experimental codebase.

### Step 1: Implement the Backend Classes

Create a new file like `my_backend.py` in this directory:

```python
"""My custom backend implementation."""

import typing as t
from .base import ProjectBackend, DatasetBackend
from ragas_experimental.model.pydantic_model import ExtendedPydanticBaseModel as BaseModel


class MyDatasetBackend(DatasetBackend):
    """Dataset backend for my storage system."""
    
    def __init__(self, connection_params: str, dataset_info: dict):
        self.connection_params = connection_params
        self.dataset_info = dataset_info
        self.dataset = None
    
    def initialize(self, dataset):
        """Initialize with dataset instance."""
        self.dataset = dataset
        # Setup storage connection, create tables/files, etc.
    
    def get_column_mapping(self, model):
        """Map model fields to storage columns."""
        # Return mapping between pydantic model fields and storage columns
        return {field: field for field in model.__annotations__.keys()}
    
    def load_entries(self, model_class):
        """Load all entries from storage."""
        # Connect to your storage and return list of model instances
        return []
    
    def append_entry(self, entry):
        """Add new entry and return its ID."""
        # Add entry to storage and return unique identifier
        return "entry_id"
    
    def update_entry(self, entry):
        """Update existing entry."""
        # Update entry in storage based on entry._row_id
        pass
    
    def delete_entry(self, entry_id):
        """Delete entry by ID."""
        # Remove entry from storage
        pass
    
    def get_entry_by_field(self, field_name: str, field_value: t.Any, model_class):
        """Find entry by field value."""
        # Query storage and return matching entry or None
        return None


class MyProjectBackend(ProjectBackend):
    """Project backend for my storage system."""
    
    def __init__(self, connection_string: str, **kwargs):
        self.connection_string = connection_string
        self.project_id = None
        # Store any additional config from **kwargs
    
    def initialize(self, project_id: str, **kwargs):
        """Initialize with project ID."""
        self.project_id = project_id
        # Setup project-level storage, create directories/schemas, etc.
    
    def create_dataset(self, name: str, model: t.Type[BaseModel]) -> str:
        """Create new dataset and return ID."""
        # Create dataset in your storage system
        dataset_id = f"dataset_{name}"
        return dataset_id
    
    def create_experiment(self, name: str, model: t.Type[BaseModel]) -> str:
        """Create new experiment and return ID."""
        # Create experiment in your storage system  
        experiment_id = f"experiment_{name}"
        return experiment_id
    
    def list_datasets(self) -> t.List[t.Dict]:
        """List all datasets."""
        # Query your storage and return list of dataset info
        return [{"id": "dataset_1", "name": "example"}]
    
    def list_experiments(self) -> t.List[t.Dict]:
        """List all experiments."""
        # Query your storage and return list of experiment info
        return [{"id": "experiment_1", "name": "example"}]
    
    def get_dataset_backend(self, dataset_id: str, name: str, model: t.Type[BaseModel]) -> DatasetBackend:
        """Get DatasetBackend for specific dataset."""
        return MyDatasetBackend(
            connection_params=self.connection_string,
            dataset_info={"id": dataset_id, "name": name}
        )
    
    def get_experiment_backend(self, experiment_id: str, name: str, model: t.Type[BaseModel]) -> DatasetBackend:
        """Get DatasetBackend for specific experiment."""
        return MyDatasetBackend(
            connection_params=self.connection_string,
            dataset_info={"id": experiment_id, "name": name}
        )
    
    def get_dataset_by_name(self, name: str, model: t.Type[BaseModel]) -> t.Tuple[str, DatasetBackend]:
        """Get dataset ID and backend by name."""
        # Query your storage to find dataset by name
        dataset_id = f"found_{name}"
        backend = self.get_dataset_backend(dataset_id, name, model)
        return dataset_id, backend
    
    def get_experiment_by_name(self, name: str, model: t.Type[BaseModel]) -> t.Tuple[str, DatasetBackend]:
        """Get experiment ID and backend by name."""
        # Query your storage to find experiment by name
        experiment_id = f"found_{name}"
        backend = self.get_experiment_backend(experiment_id, name, model)
        return experiment_id, backend
```

### Step 2: Register the Backend

Update `registry.py` to include your backend in the built-in backends:

```python
# In _register_builtin_backends method
def _register_builtin_backends(self) -> None:
    """Register the built-in backends."""
    try:
        from .local_csv import LocalCSVProjectBackend
        self.register_backend("local_csv", LocalCSVProjectBackend, aliases=["local"])
        
        from .platform import PlatformProjectBackend
        self.register_backend("platform", PlatformProjectBackend, aliases=["ragas_app"])
        
        # Add your backend here
        from .my_backend import MyProjectBackend
        self.register_backend("my_storage", MyProjectBackend, aliases=["custom"])
        
    except ImportError as e:
        logger.warning(f"Failed to import built-in backend: {e}")
```

### Step 3: Add Entry Point Configuration

Update `experimental/pyproject.toml` to include your backend:

```toml
[project.entry-points."ragas.backends"]
local_csv = "ragas_experimental.project.backends.local_csv:LocalCSVProjectBackend"
platform = "ragas_experimental.project.backends.platform:PlatformProjectBackend"
my_storage = "ragas_experimental.project.backends.my_backend:MyProjectBackend"
```

### Step 4: Update Exports

Add your backend to `__init__.py`:

```python
# Import concrete backends for backward compatibility
from .local_csv import LocalCSVProjectBackend
from .platform import PlatformProjectBackend
from .my_backend import MyProjectBackend  # Add this

__all__ = [
    "ProjectBackend",
    "DatasetBackend",
    # ... other exports ...
    "MyProjectBackend",  # Add this
]
```

### Step 5: Write Tests

Create `test_my_backend.py`:

```python
"""Tests for my custom backend."""

import pytest
import tempfile
from ragas_experimental.project.backends.my_backend import MyProjectBackend, MyDatasetBackend


def test_my_backend_creation():
    """Test backend can be created."""
    backend = MyProjectBackend(connection_string="test://connection")
    assert backend.connection_string == "test://connection"


def test_my_backend_integration():
    """Test backend works with project system."""
    from ragas_experimental.project import create_project
    
    project = create_project(
        name="test_project",
        backend="my_storage",
        connection_string="test://connection"
    )
    
    assert project.name == "test_project"
    # Add more integration tests...
```

## Section 2: Creating Pip-Installable Backend Plugins

Create a separate Python package that provides a backend plugin.

### Plugin Package Structure

```
ragas-sqlite-backend/
├── pyproject.toml
├── README.md
├── src/
│   └── ragas_sqlite_backend/
│       ├── __init__.py
│       ├── backend.py
│       └── dataset.py
└── tests/
    └── test_sqlite_backend.py
```

### Step 1: Create the Plugin Package

**pyproject.toml**:
```toml
[build-system]
requires = ["setuptools>=64", "setuptools_scm>=8"]
build-backend = "setuptools.build_meta"

[project]
name = "ragas-sqlite-backend"
version = "0.1.0"
description = "SQLite backend for Ragas experimental projects"
authors = [{name = "Your Name", email = "your.email@example.com"}]
requires-python = ">=3.9"
dependencies = [
    "ragas_experimental",  # Depend on the main package
    "sqlite3",  # If not in stdlib
]

# Define the entry point for backend discovery
[project.entry-points."ragas.backends"]
sqlite = "ragas_sqlite_backend.backend:SQLiteProjectBackend"

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio"]
```

**src/ragas_sqlite_backend/backend.py**:
```python
"""SQLite backend implementation."""

import sqlite3
import typing as t
from pathlib import Path

# Import from the main ragas_experimental package
from ragas_experimental.project.backends.base import ProjectBackend, DatasetBackend
from ragas_experimental.model.pydantic_model import ExtendedPydanticBaseModel as BaseModel


class SQLiteDatasetBackend(DatasetBackend):
    """SQLite implementation of DatasetBackend."""
    
    def __init__(self, db_path: str, table_name: str):
        self.db_path = db_path
        self.table_name = table_name
        self.dataset = None
    
    def initialize(self, dataset):
        """Initialize with dataset and create table."""
        self.dataset = dataset
        self._create_table_if_not_exists()
    
    def _create_table_if_not_exists(self):
        """Create SQLite table based on model schema."""
        with sqlite3.connect(self.db_path) as conn:
            # Create table based on model fields
            model_fields = self.dataset.model.__annotations__
            
            columns = ["_row_id TEXT PRIMARY KEY"]
            for field_name, field_type in model_fields.items():
                sql_type = self._python_to_sql_type(field_type)
                columns.append(f"{field_name} {sql_type}")
            
            create_sql = f"CREATE TABLE IF NOT EXISTS {self.table_name} ({', '.join(columns)})"
            conn.execute(create_sql)
    
    def _python_to_sql_type(self, python_type):
        """Convert Python type to SQLite type."""
        type_mapping = {
            str: "TEXT",
            int: "INTEGER", 
            float: "REAL",
            bool: "INTEGER",
        }
        return type_mapping.get(python_type, "TEXT")
    
    # Implement all other abstract methods...
    def get_column_mapping(self, model):
        return {field: field for field in model.__annotations__.keys()}
    
    def load_entries(self, model_class):
        # Implement SQLite loading logic
        return []
    
    def append_entry(self, entry):
        # Implement SQLite insertion logic
        return "new_entry_id"
    
    # ... implement other required methods


class SQLiteProjectBackend(ProjectBackend):
    """SQLite implementation of ProjectBackend."""
    
    def __init__(self, db_path: str = None, **kwargs):
        self.db_path = db_path or "ragas_project.db"
        self.project_id = None
    
    def initialize(self, project_id: str, **kwargs):
        """Initialize SQLite database for project."""
        self.project_id = project_id
        
        # Create database file and project metadata table
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Create metadata tables
            conn.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    project_id TEXT,
                    name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (id)
                )
            """)
            
            # Insert project if not exists
            conn.execute(
                "INSERT OR IGNORE INTO projects (id, name) VALUES (?, ?)",
                (project_id, project_id)
            )
    
    # Implement all abstract methods...
    def create_dataset(self, name: str, model: t.Type[BaseModel]) -> str:
        # Implement dataset creation in SQLite
        dataset_id = f"dataset_{name}_{self.project_id}"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO datasets (id, project_id, name) VALUES (?, ?, ?)",
                (dataset_id, self.project_id, name)
            )
        
        return dataset_id
    
    def get_dataset_backend(self, dataset_id: str, name: str, model: t.Type[BaseModel]) -> DatasetBackend:
        """Return SQLite dataset backend."""
        table_name = f"data_{dataset_id}"
        return SQLiteDatasetBackend(self.db_path, table_name)
    
    # ... implement other required methods
```

**src/ragas_sqlite_backend/__init__.py**:
```python
"""SQLite backend plugin for Ragas experimental."""

from .backend import SQLiteProjectBackend, SQLiteDatasetBackend

__all__ = ["SQLiteProjectBackend", "SQLiteDatasetBackend"]
```

### Step 2: Publish the Plugin

1. **Build the package**:
   ```bash
   pip install build
   python -m build
   ```

2. **Upload to PyPI** (optional):
   ```bash
   pip install twine
   twine upload dist/*
   ```

3. **Install and test**:
   ```bash
   pip install ragas-sqlite-backend
   
   # The backend should now be automatically discovered
   python -c "from ragas_experimental.project import list_backends; print(list_backends())"
   # Should include 'sqlite' in the output
   ```

### Step 3: Use the Plugin

Once installed, users can use your backend:

```python
from ragas_experimental.project import create_project

# Use your plugin backend
project = create_project(
    name="my_sqlite_project",
    backend="sqlite",  # Your plugin's entry point name
    db_path="/path/to/database.db"
)

# Backend works seamlessly with the rest of the system
dataset = project.create_dataset("my_data", MyDataModel)
dataset.add_entries([...])
```

## Best Practices

### Error Handling
- Use proper logging: `import logging; logger = logging.getLogger(__name__)`
- Handle connection failures gracefully
- Provide meaningful error messages

### Performance
- Implement connection pooling for database backends
- Use batch operations when possible
- Consider caching for frequently accessed data

### Testing
- Test both ProjectBackend and DatasetBackend separately
- Include integration tests with the Project class
- Test error conditions and edge cases
- Use temporary storage for tests (tempfile, in-memory DBs)

### Documentation
- Document all configuration parameters
- Provide usage examples
- Include troubleshooting guides

### Configuration
- Accept configuration through constructor kwargs
- Support environment variables for sensitive data
- Provide sensible defaults

## Common Patterns

### Connection Management
```python
class MyBackend(ProjectBackend):
    def __init__(self, connection_string: str, **kwargs):
        self.connection_string = connection_string
        self._connection = None
    
    def _get_connection(self):
        """Lazy connection initialization."""
        if self._connection is None:
            self._connection = create_connection(self.connection_string)
        return self._connection
```

### ID Generation
```python
from ragas_experimental.project.utils import create_nano_id

def create_dataset(self, name: str, model):
    dataset_id = create_nano_id()  # Creates unique short ID
    # ... rest of implementation
```

### Model Validation
```python
def append_entry(self, entry):
    # Validate entry is correct model type
    if not isinstance(entry, self.dataset.model):
        raise ValueError(f"Entry must be instance of {self.dataset.model}")
    
    # Add to storage...
```

For more examples, see the existing `local_csv.py` and `platform.py` implementations in this directory.