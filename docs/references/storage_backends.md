# Storage Backends

Storage backends are the persistence layer underneath Ragas datasets and
experiments. Each backend implements the same small interface — load,
save, list — so evaluation pipelines can switch between local files,
in-memory state, or remote storage without changing application code.

This page covers the four built-in backends, the `BackendRegistry` that
discovers them, and the trade-offs between them.

- [Overview](#overview)
- [Built-in backends](#built-in-backends)
- [Choosing a backend](#choosing-a-backend)
- [Backend registry](#backend-registry)
- [Examples](#examples)
- [Limitations and notes](#limitations-and-notes)

## Overview

A backend is any class that subclasses `BaseBackend` (defined in
`src/ragas/backends/base.py`) and provides four operations per resource
type (`datasets` and `experiments`):

| Operation        | Behaviour |
|------------------|-----------|
| `load_<resource>`| Return rows; raise `FileNotFoundError` when missing |
| `save_<resource>`| Persist rows; create directories as needed |
| `list_<resource>`| Return the names of every stored row |
| `delete_<resource>`| Remove a stored row (used for cleanup) |

Backends are stateless with respect to a single resource; they accept a
name (string identifier) plus the row payload. This makes it cheap to
swap one backend for another between runs.

The four built-in backends are:

| Backend               | Storage | Best for |
|-----------------------|---------|----------|
| `LocalCSVBackend`     | CSV files on disk | Simple tabular data, spreadsheet-style inspection |
| `LocalJSONLBackend`   | JSONL files on disk | Nested structures, datetime fidelity, larger sets |
| `InMemoryBackend`     | Process memory | Tests, ephemeral notebooks, no-disk experiments |
| `GDriveBackend`       | Google Drive via the Drive API | Team-shared persistence, audit-friendly storage |

## Built-in backends

### `LocalCSVBackend`

Persists rows as CSV files under `<root_dir>/datasets/<name>.csv` and
`<root_dir>/experiments/<name>.csv`. Best when every row is a flat
mapping of primitive values and you want to inspect or hand-edit the
file outside Python.

```python
from ragas.backends import LocalCSVBackend

backend = LocalCSVBackend(root_dir="./ragas_store")
backend.save_dataset("qa_smoke", [{"user_input": "Q", "reference": "A"}])
print(backend.list_datasets())  # ['qa_smoke']
```

### `LocalJSONLBackend`

Persists rows as one JSON object per line under
`<root_dir>/datasets/<name>.jsonl` and
`<root_dir>/experiments/<name>.jsonl`. Preserves nested dicts, lists, and
`datetime` values through ISO encoding, and tolerates malformed lines
with a warning instead of failing the whole load.

```python
from ragas.backends import LocalJSONLBackend

backend = LocalJSONLBackend(root_dir="./ragas_store")
backend.save_experiment(
    "exp_2026_07",
    [{"metric": "faithfulness", "value": 0.87, "ts": "2026-07-04T12:00:00"}],
)
```

### `InMemoryBackend`

Stores everything in a dict. Use it in unit tests, short-lived
notebooks, or anywhere persistence is undesirable. Nothing survives the
process — by design.

```python
from ragas.backends import InMemoryBackend

backend = InMemoryBackend()
backend.save_dataset("ephemeral", rows)
assert backend.load_dataset("ephemeral") == rows
```

### `GDriveBackend`

Persists datasets and experiments as Google Sheets in a configured
Google Drive folder. Useful when a team needs a shared, auditable record
of evaluation runs without standing up a database.

```python
from ragas.backends import GDriveBackend

backend = GDriveBackend(
    folder_id="<google-drive-folder-id>",
    credentials_path="./service-account.json",
)
backend.save_dataset("shared_qa_set", rows)
```

See the module docstring in `src/ragas/backends/gdrive_backend.py` for
the credentials and folder-id setup.

## Choosing a backend

| If your data is …                                | Pick                       |
|--------------------------------------------------|----------------------------|
| Flat rows of primitives, and you want CSVs        | `LocalCSVBackend`          |
| Nested dicts, lists, or datetimes, on disk       | `LocalJSONLBackend`        |
| Throwaway, in-process, or under a unit test      | `InMemoryBackend`          |
| Shared with collaborators via Google Drive        | `GDriveBackend`            |

For most evaluation pipelines, `LocalJSONLBackend` is the default — it
keeps nested structures intact and round-trips datetimes without losing
type fidelity.

## Backend registry

`BackendRegistry` is a singleton that maps a name (string) to a backend
class. The discovery entry point is `ragas.backends.get_registry()`.

```python
from ragas.backends import get_registry

registry = get_registry()
print(registry.list_all_names())
# {'local_csv': ['local_csv'], 'local_jsonl': ['local_jsonl'], ...}
```

### Look up by name

```python
from ragas.backends import get_registry

BackendCls = get_registry()["local_jsonl"]
backend = BackendCls(root_dir="./ragas_store")
```

### Register aliases

```python
from ragas.backends import get_registry

get_registry().register_aliases("local_jsonl", ["jsonl", "json"])
# Later lookups work with any of the registered names.
get_registry()["jsonl"]
```

### Register a custom backend

Any subclass of `BaseBackend` can be registered so it is discoverable
under a string name:

```python
from ragas.backends import BaseBackend, get_registry


class S3Backend(BaseBackend):
    def __init__(self, bucket: str):
        self.bucket = bucket

    def load_dataset(self, name): ...
    def save_dataset(self, name, data, data_model=None): ...
    # implement the remaining methods


get_registry()["s3"] = S3Backend
backend = get_registry().create_backend("s3", bucket="my-bucket")
```

Third-party packages can also expose backends through the
`ragas.backends` entry-point group, in which case `BackendRegistry`
discovers them on first use.

## Examples

### Round-trip a small dataset with `LocalJSONLBackend`

```python
from ragas.backends import LocalJSONLBackend

backend = LocalJSONLBackend(root_dir="./local_ragas_store")
rows = [
    {"user_input": "What is 2+2?", "reference": "4"},
    {"user_input": "Capital of France?", "reference": "Paris"},
]
backend.save_dataset("math_smoke", rows)
loaded = backend.load_dataset("math_smoke")
assert loaded == rows
```

### Run an evaluation that writes to Google Drive

```python
from ragas.backends import GDriveBackend

backend = GDriveBackend(
    folder_id="<google-drive-folder-id>",
    credentials_path="./service-account.json",
)

# Use backend as the persistence layer for your experiment / dataset.
# See concepts/datasets.md and concepts/experimentation.md for end-to-end
# examples that wire a backend into EvaluationDataset and Experiment.
```

### Look up a custom backend by alias

```python
from ragas.backends import get_registry

BackendCls = get_registry()["jsonl"]  # alias for local_jsonl
backend = BackendCls(root_dir="./ragas_store")
```

## Limitations and notes

- **`LocalCSVBackend`** flattens every value to a string. Lists, dicts,
  and datetimes lose their native types on round-trip. Reach for
  `LocalJSONLBackend` instead when row shape is non-trivial.
- **`LocalJSONLBackend`** keeps the file open in append mode while
  writing and tolerates malformed lines on load (they're skipped with a
  warning). It is **not** a transactional store — concurrent writers
  to the same `<name>.jsonl` can race.
- **`InMemoryBackend`** does not persist anything. It is intentional
  that nothing survives across processes; do not use it for anything
  you intend to keep.
- **`GDriveBackend`** requires a service-account JSON and a folder the
  service account can write to. Sharing happens via Drive permissions,
  not via Ragas.
- All backends treat `name` as a **logical identifier**, not a path.
  Pass something like `"qa_smoke"` — do not pass `../escape` or
  absolute paths; the build rejects and the run raises an error.

For caching, see [Cache](cache.md); backends and the cache are
independent layers that compose.

## See also

- [Datasets](../concepts/datasets.md)
- [Experimentation](../concepts/experimentation.md)
- [Cache](cache.md)
- [Evaluation Dataset](../concepts/components/eval_dataset.md)