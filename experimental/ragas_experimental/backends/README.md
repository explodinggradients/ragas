# Ragas Backends

Backends store your project data (datasets/experiments) in different places: local files, databases, cloud APIs. You implement 2 classes: `ProjectBackend` (manages projects) and `DataTableBackend` (handles data operations).

```
Project → ProjectBackend → DataTableBackend → Storage
```

## Current State

**Available Backends:**
- `local/csv` - Local CSV files
- `ragas/app` - Ragas cloud platform  
- `box/csv` - Box cloud storage

**Import Path:** `ragas_experimental.backends`

**Core Classes:**
- `ProjectBackend` - Project-level operations (create datasets/experiments)
- `DataTableBackend` - Data operations (read/write entries) 
- `DataTable` - Base class for `Dataset` and `Experiment`

## Learning Roadmap

Follow this path to build your own backend:

```
□ 1. Understand: Read local_csv.py (simplest example)
□ 2. Explore: Study base.py abstract methods  
□ 3. Practice: Modify LocalCSVBackend to add logging
□ 4. Build: Create your own backend following the pattern
□ 5. Advanced: Study ragas_app.py for API/async patterns
□ 6. Package: Create plugin (see Plugin Development)
```

## Quick Usage

**Using existing backends:**
```python
from ragas_experimental.project import Project

# Local CSV
project = Project.create("my_project", "local/csv", root_dir="./data")

# Ragas platform  
project = Project.create("my_project", "ragas/app", api_key="your_key")
```

**Basic backend structure:**
```python
from ragas_experimental.backends.base import ProjectBackend, DataTableBackend

class MyProjectBackend(ProjectBackend):
    def create_dataset(self, name, model): 
        # Create storage space for dataset
        pass
    
class MyDataTableBackend(DataTableBackend):
    def load_entries(self, model_class):
        # Load entries from storage
        pass
```

## Essential Methods

**ProjectBackend** (project management):
- `create_dataset()` / `create_experiment()` - Create storage
- `get_dataset_backend()` / `get_experiment_backend()` - Get data handler
- `list_datasets()` / `list_experiments()` - List existing

**DataTableBackend** (data operations):
- `initialize()` - Setup with dataset instance
- `load_entries()` - Load all entries
- `append_entry()` - Add new entry
- `update_entry()` / `delete_entry()` - Modify entries

See `base.py` for complete interface.

## Learn from Examples

**Start here:**
- `local_csv.py` - File-based storage, easiest to understand
- `config.py` - Configuration patterns

**Advanced patterns:**
- `ragas_app.py` - API calls, async, error handling
- `box_csv.py` - Cloud storage, authentication
- `registry.py` - Backend discovery system

## Quick Development

**1. Copy template:**
```bash
cp local_csv.py my_backend.py
```

**2. Replace CSV logic with your storage**

**3. Register backend:**
```python
# In registry.py _register_builtin_backends()
from .my_backend import MyProjectBackend
self.register_backend("my_storage", MyProjectBackend)
```

**4. Test:**
```python
project = Project.create("test", "my_storage")
```

## Plugin Development

**Create separate package:**
```
my-backend-plugin/
├── pyproject.toml
├── src/my_backend/
│   ├── __init__.py
│   └── backend.py
└── tests/
```

**Entry point in pyproject.toml:**
```toml
[project.entry-points."ragas.backends"]
my_storage = "my_backend.backend:MyProjectBackend"
```

**Install and use:**
```bash
pip install my-backend-plugin
python -c "from ragas_experimental.project import Project; Project.create('test', 'my_storage')"
```

## Common Patterns

**ID Generation:**
```python
from .utils import create_nano_id
dataset_id = create_nano_id()
```

**Error Handling:**
```python
try:
    # Storage operation
except ConnectionError:
    # Handle gracefully
```

**Testing:**
```python
def test_my_backend():
    backend = MyProjectBackend()
    backend.initialize("test_project")
    dataset_id = backend.create_dataset("test", MyModel)
    assert dataset_id
```

## Troubleshooting

**Backend not found?** Check registry with:
```python
from ragas_experimental.backends import list_backends
print(list_backends())
```

**Entries not loading?** Verify:
- `initialize()` called before other methods
- `load_entries()` returns list of model instances
- Entry `_row_id` attributes set correctly

**Need help?** Study existing backends - they handle most common patterns.

## Configuration Examples

**Local CSV:**
```python
from ragas_experimental.backends import LocalCSVConfig
config = LocalCSVConfig(root_dir="/path/to/data")
```

**Ragas App:**
```python  
from ragas_experimental.backends import RagasAppConfig
config = RagasAppConfig(api_key="key", api_url="https://api.ragas.io")
```

**Box CSV:**
```python
from ragas_experimental.backends import BoxCSVConfig
config = BoxCSVConfig(client=authenticated_box_client)
```

---

**Next Steps:** Start with modifying `local_csv.py`, then build your own following the same patterns.
