"""A python list like object that contains your evaluation data."""

__all__ = [
    "BaseModelType",
    "Dataset",
]

import typing as t

try:
    import pandas as pd
except ImportError:
    pd = None

from ragas_experimental.model.pydantic_model import (
    ExtendedPydanticBaseModel as BaseModel,
)

from .backends.ragas_api_client import RagasApiClient
from .project.backends import (
    LocalCSVProjectBackend,
    PlatformProjectBackend,
)
from .typing import SUPPORTED_BACKENDS

BaseModelType = t.TypeVar("BaseModelType", bound=BaseModel)


class Dataset(t.Generic[BaseModelType]):
    """A list-like interface for managing dataset entries with backend synchronization.

    This class behaves like a Python list while synchronizing operations with the
    chosen backend (Ragas API or local filesystem).
    """

    def __init__(
        self,
        name: str,
        model: t.Type[BaseModel],
        project_id: str,
        dataset_id: str,
        datatable_type: t.Literal["datasets", "experiments"],
        ragas_api_client: t.Optional[RagasApiClient] = None,
        backend: SUPPORTED_BACKENDS = "local/csv",
        local_root_dir: t.Optional[str] = None,
    ):
        """Initialize a Dataset with the specified backend.

        Args:
            name: The name of the dataset
            model: The Pydantic model class for entries
            project_id: The ID of the parent project
            dataset_id: The ID of this dataset
            datatable_type: Whether this is for "datasets" or "experiments"
            ragas_api_client: Required for ragas/app backend
            backend: The storage backend to use (ragas/app or local/csv)
            local_root_dir: Required for local backend
        """
        # Store basic properties
        self.name = name
        self.model = model
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.backend_type = backend
        self.datatable_type = datatable_type
        self._entries: t.List[BaseModelType] = []

        # Create the appropriate backend using the project backend system
        if backend == "ragas/app":
            if ragas_api_client is None:
                raise ValueError("ragas_api_client is required for ragas/app backend")

            # Create a platform project backend and get dataset backend from it
            project_backend = PlatformProjectBackend(ragas_api_client)
            project_backend.initialize(project_id)

            if datatable_type == "datasets":
                self._backend = project_backend.get_dataset_backend(
                    dataset_id, name, model
                )
            else:  # experiments
                self._backend = project_backend.get_experiment_backend(
                    dataset_id, name, model
                )

        elif backend == "local/csv":
            if local_root_dir is None:
                raise ValueError("local_root_dir is required for local/csv backend")

            # Create a local CSV project backend and get dataset backend from it
            project_backend = LocalCSVProjectBackend(local_root_dir)
            project_backend.initialize(project_id)

            if datatable_type == "datasets":
                self._backend = project_backend.get_dataset_backend(
                    dataset_id, name, model
                )
            else:  # experiments
                self._backend = project_backend.get_experiment_backend(
                    dataset_id, name, model
                )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        # Initialize the backend with this dataset
        self._backend.initialize(self)

        # Initialize column mapping if it doesn't exist yet
        if not hasattr(self.model, "__column_mapping__"):
            self.model.__column_mapping__ = {}

        # Get column mappings from backend and update the model's mapping
        column_mapping = self._backend.get_column_mapping(model)

        # Update the model's column mapping
        for field_name, column_id in column_mapping.items():
            self.model.__column_mapping__[field_name] = column_id

    def __getitem__(
        self, key: t.Union[int, slice]
    ) -> t.Union[BaseModelType, "Dataset[BaseModelType]"]:
        """Get an entry by index or slice."""
        if isinstance(key, slice):
            # Create a shallow copy of the dataset
            new_dataset = object.__new__(type(self))

            # Copy all attributes
            new_dataset.name = self.name
            new_dataset.model = self.model
            new_dataset.project_id = self.project_id
            new_dataset.dataset_id = self.dataset_id
            new_dataset.backend_type = self.backend_type
            new_dataset.datatable_type = self.datatable_type

            # Share the same backend reference
            new_dataset._backend = self._backend

            # Set the entries to the sliced entries
            new_dataset._entries = self._entries[key]

            return new_dataset
        else:
            return self._entries[key]

    def __setitem__(self, index: int, entry: BaseModelType) -> None:
        """Update an entry at the given index and sync to backend."""
        if not isinstance(entry, self.model):
            raise TypeError(f"Entry must be an instance of {self.model.__name__}")

        # Get existing entry to get its ID
        existing = self._entries[index]
        if hasattr(existing, "_row_id") and existing._row_id:
            entry._row_id = existing._row_id

        # Update in backend
        self._backend.update_entry(entry)

        # Update local cache
        self._entries[index] = entry

    def __repr__(self) -> str:
        """String representation of the dataset."""
        return (
            f"Dataset(name='{self.name}', model={self.model.__name__}, len={len(self)})"
        )

    def __len__(self) -> int:
        """Get the number of entries in the dataset."""
        return len(self._entries)

    def __iter__(self) -> t.Iterator[BaseModelType]:
        """Iterate over the entries in the dataset."""
        return iter(self._entries)

    def get_fields_by_type(self, target_type: t.Any) -> t.List[str]:
        """Get field names that match the given type.

        Handles complex types like Union, Optional, etc. using typing helpers.

        Args:
            target_type: The type to match against

        Returns:
            List of field names with matching type
        """
        return_fields = []
        for field_name, field_info in self.model.model_fields.items():
            annotation = field_info.annotation

            # Handle direct type match
            if annotation == target_type:
                return_fields.append(field_name)
                continue

            # Handle complex types like Union, Optional, etc.
            origin = t.get_origin(annotation)
            args = t.get_args(annotation)

            # Check for Optional[target_type] or Union[target_type, None]
            if origin is t.Union and target_type in args:
                return_fields.append(field_name)
            # Check for List[target_type], Dict[_, target_type], etc.
            elif origin and args and any(arg == target_type for arg in args):
                return_fields.append(field_name)

        return return_fields

    def append(self, entry: BaseModelType) -> None:
        """Add a new entry to the dataset and sync to backend.

        Args:
            entry: The entry to add to the dataset
        """
        if not isinstance(entry, self.model):
            raise TypeError(f"Entry must be an instance of {self.model.__name__}")

        # Add to backend and get ID
        row_id = self._backend.append_entry(entry)

        # Store the ID
        entry._row_id = row_id

        # Add to local cache
        self._entries.append(entry)

    def pop(self, index: int = -1) -> BaseModelType:
        """Remove and return entry at index, sync deletion to backend.

        Args:
            index: The index of the entry to remove (default: -1, the last entry)

        Returns:
            The removed entry
        """
        # Get the entry
        entry = self._entries[index]

        # Get the row ID
        row_id = getattr(entry, "_row_id", None)
        if row_id is None:
            raise ValueError(
                "Entry has no row ID. This likely means it was not added or synced to the dataset."
            )

        # Delete from backend
        self._backend.delete_entry(row_id)

        # Remove from local cache
        return self._entries.pop(index)

    def load(self) -> None:
        """Load all entries from the backend."""
        # Get entries from backend
        self._entries = self._backend.load_entries(self.model)

    def load_as_dicts(self) -> t.List[t.Dict]:
        """Load all entries as dictionaries.

        Returns:
            List of dictionaries representing the entries
        """
        # Make sure we have entries
        if not self._entries:
            self.load()

        # Convert to dictionaries
        return [entry.model_dump() for entry in self._entries]

    def to_pandas(self) -> "pd.DataFrame":
        """Convert dataset to pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing all entries

        Raises:
            ImportError: If pandas is not installed
        """
        if pd is None:
            raise ImportError(
                "pandas is required for to_pandas(). Install with: pip install pandas "
                "or pip install ragas_experimental[all]"
            )

        # Make sure we have data
        if not self._entries:
            self.load()

        # Convert entries to dictionaries
        data = [entry.model_dump() for entry in self._entries]
        return pd.DataFrame(data)

    def save(self, item: BaseModelType) -> None:
        """Save changes to an item to the backend.

        Args:
            item: The item to save
        """
        if not isinstance(item, self.model):
            raise TypeError(f"Item must be an instance of {self.model.__name__}")

        # Check if the item has a row ID
        if not hasattr(item, "_row_id") or not item._row_id:
            # Try to find it in our entries by matching
            for i, entry in enumerate(self._entries):
                if id(entry) == id(item):  # Check if it's the same object
                    if hasattr(entry, "_row_id") and entry._row_id:
                        item._row_id = entry._row_id
                        break

        if not hasattr(item, "_row_id") or not item._row_id:
            raise ValueError(
                "Cannot save: item is not from this dataset or was not properly synced"
            )

        # Update in backend
        self._backend.update_entry(item)

        # Update in local cache if needed
        self._update_local_entry(item)

    def _update_local_entry(self, item: BaseModelType) -> None:
        """Update an entry in the local cache.

        Args:
            item: The item to update
        """
        for i, entry in enumerate(self._entries):
            if (
                hasattr(entry, "_row_id")
                and hasattr(item, "_row_id")
                and entry._row_id == item._row_id
            ):
                # If it's not the same object, update our copy
                if id(entry) != id(item):
                    self._entries[i] = item
                break

    def get(
        self, field_value: t.Any, field_name: str = "_row_id"
    ) -> t.Optional[BaseModelType]:
        """Get an entry by field value.

        Args:
            field_value: The value to match
            field_name: The field to match against (default: "_row_id")

        Returns:
            The matching model instance or None if not found
        """
        # Check if we need to load entries
        if not self._entries:
            self.load()

        # Search in local entries first
        for entry in self._entries:
            if hasattr(entry, field_name) and getattr(entry, field_name) == field_value:
                return entry

        # If not found, try to get from backend
        if field_name == "_row_id":
            # Special case for row IDs
            for entry in self._entries:
                if hasattr(entry, "_row_id") and entry._row_id == field_value:
                    return entry
        else:
            # Use backend to search
            return self._backend.get_entry_by_field(field_name, field_value, self.model)

        return None

    def train_test_split(
        self, test_size: float = 0.2, random_state: t.Optional[int] = None
    ) -> t.Tuple["Dataset[BaseModelType]", "Dataset[BaseModelType]"]:
        """Split the dataset into training and testing sets.

        Args:
            test_size: Proportion of the dataset to include in the test split (default: 0.2)
            random_state: Random seed for reproducibility (default: None)
        Returns:
            A tuple of two Datasets: (train_dataset, test_dataset)
        """
        if not self._entries:
            self.load()

        # Shuffle entries if random_state is set
        if random_state is not None:
            import random

            random.seed(random_state)
            random.shuffle(self._entries)

        # Calculate split index
        split_index = int(len(self._entries) * (1 - test_size))

        # Create new dataset instances without full initialization
        train_dataset = object.__new__(type(self))
        test_dataset = object.__new__(type(self))

        # Copy essential attributes
        for dataset in [train_dataset, test_dataset]:
            dataset.model = self.model
            dataset.project_id = self.project_id
            dataset._backend = self._backend
            dataset.backend_type = self.backend_type
            dataset.datatable_type = self.datatable_type

        # Set specific attributes for each dataset
        train_dataset.name = f"{self.name}_train"
        train_dataset.dataset_id = f"{self.dataset_id}_train"

        test_dataset.name = f"{self.name}_test"
        test_dataset.dataset_id = f"{self.dataset_id}_test"

        # Assign entries to the new datasets
        train_dataset._entries = self._entries[:split_index]
        test_dataset._entries = self._entries[split_index:]

        return train_dataset, test_dataset
