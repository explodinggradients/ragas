"""A python list like object that contains your evaluation data."""

__all__ = [
    "BaseModelType",
    "DataTable",
    "Dataset",
]

import typing as t
from typing import overload, Literal, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

from ragas_experimental.model.pydantic_model import (
    ExtendedPydanticBaseModel as BaseModel,
)

from .backends import RagasApiClient
from .backends import create_project_backend, DataTableBackend
from .typing import SUPPORTED_BACKENDS

# Type-only imports
if t.TYPE_CHECKING:
    from .project.core import Project

BaseModelType = t.TypeVar("BaseModelType", bound=BaseModel)


class DataTable(t.Generic[BaseModelType]):
    """A list-like interface for managing datatable entries with backend synchronization.

    This class behaves like a Python list while synchronizing operations with the
    chosen backend (Ragas API or local filesystem). Base class for Dataset and Experiment.
    """

    # Type-safe overloads for dataset creation
    @overload
    @classmethod
    def create(
        cls,
        name: str,
        model: t.Type[BaseModel],
        project: "Project",
        dataset_type: Literal["datasets"] = "datasets"
    ) -> "DataTable[BaseModel]": ...

    @overload
    @classmethod
    def create(
        cls,
        name: str,
        model: t.Type[BaseModel],
        project: "Project", 
        dataset_type: Literal["experiments"]
    ) -> "DataTable[BaseModel]": ...

    @classmethod
    def create(
        cls,
        name: str,
        model: t.Type[BaseModel],
        project: "Project",
        dataset_type: Literal["datasets", "experiments"] = "datasets"
    ) -> "DataTable[BaseModel]":
        """Create a new dataset with type-safe parameters.

        Args:
            name: Name of the dataset
            model: Pydantic model class for entries
            project: Project instance to create the dataset in
            dataset_type: Type of dataset ("datasets" or "experiments")

        Returns:
            Dataset: A new dataset instance

        Examples:
            >>> # Create a dataset
            >>> dataset = Dataset.create("my_data", MyModel, project)
            
            >>> # Create an experiment
            >>> experiment = Dataset.create("my_experiment", MyModel, project, "experiments")
        """
        # Use the project's backend to create the dataset
        if dataset_type == "datasets":
            dataset_id = project._backend.create_dataset(name, model)
            backend = project._backend.get_dataset_backend(dataset_id, name, model)
        else:  # experiments
            dataset_id = project._backend.create_experiment(name, model)
            backend = project._backend.get_experiment_backend(dataset_id, name, model)

        # Create the dataset with the simplified constructor
        return cls._create_with_backend(
            name=name,
            model=model,
            project_id=project.project_id,
            dataset_id=dataset_id,
            datatable_type=dataset_type,
            backend=backend
        )

    # Type-safe overloads for getting existing datasets
    @overload
    @classmethod
    def get_dataset(
        cls,
        name: str,
        model: t.Type[BaseModel],
        project: "Project",
        dataset_type: Literal["datasets"] = "datasets"
    ) -> "DataTable[BaseModel]": ...

    @overload
    @classmethod
    def get_dataset(
        cls,
        name: str,
        model: t.Type[BaseModel],
        project: "Project",
        dataset_type: Literal["experiments"]
    ) -> "DataTable[BaseModel]": ...

    @classmethod
    def get_dataset(
        cls,
        name: str,
        model: t.Type[BaseModel],
        project: "Project",
        dataset_type: Literal["datasets", "experiments"] = "datasets"
    ) -> "DataTable[BaseModel]":
        """Get an existing dataset by name with type-safe parameters.

        Args:
            name: Name of the dataset to retrieve
            model: Pydantic model class for entries
            project: Project instance containing the dataset
            dataset_type: Type of dataset ("datasets" or "experiments")

        Returns:
            Dataset: The existing dataset instance

        Examples:
            >>> # Get a dataset
            >>> dataset = Dataset.get_dataset("my_data", MyModel, project)
            
            >>> # Get an experiment  
            >>> experiment = Dataset.get_dataset("my_experiment", MyModel, project, "experiments")
        """
        # Use the project's backend to get the dataset
        if dataset_type == "datasets":
            dataset_id, _ = project._backend.get_dataset_by_name(name, model)
            backend = project._backend.get_dataset_backend(dataset_id, name, model)
        else:  # experiments
            dataset_id, _ = project._backend.get_experiment_by_name(name, model)
            backend = project._backend.get_experiment_backend(dataset_id, name, model)

        # Create the dataset with the simplified constructor
        return cls._create_with_backend(
            name=name,
            model=model,
            project_id=project.project_id,
            dataset_id=dataset_id,
            datatable_type=dataset_type,
            backend=backend
        )

    @classmethod
    def _create_with_backend(
        cls,
        name: str,
        model: t.Type[BaseModel],
        project_id: str,
        dataset_id: str,
        datatable_type: t.Literal["datasets", "experiments"],
        backend: DataTableBackend
    ) -> "DataTable[BaseModel]":
        """Internal helper to create a dataset with a backend instance.
        
        Args:
            name: Dataset name
            model: Pydantic model class
            project_id: Project ID
            dataset_id: Dataset ID
            datatable_type: Dataset or experiment type
            backend: Backend instance
            
        Returns:
            DataTable: New datatable instance
        """
        # Create the instance without calling __init__
        instance = cls.__new__(cls)
        
        # Set basic properties
        instance.name = name
        instance.model = model
        instance.project_id = project_id
        instance.dataset_id = dataset_id
        instance.backend_type = getattr(backend, 'backend_type', 'unknown')
        instance.datatable_type = datatable_type
        instance._entries = []
        instance._backend = backend
        
        # Initialize the backend with this dataset
        instance._backend.initialize(instance)
        
        # Initialize column mapping if it doesn't exist yet
        if not hasattr(instance.model, "__column_mapping__"):
            instance.model.__column_mapping__ = {}

        # Get column mappings from backend and update the model's mapping
        column_mapping = instance._backend.get_column_mapping(model)
        for field_name, column_id in column_mapping.items():
            instance.model.__column_mapping__[field_name] = column_id
            
        return instance

    def __init__(
        self,
        name: str,
        model: t.Type[BaseModel],
        project_id: str,
        dataset_id: str,
        datatable_type: t.Literal["datasets", "experiments"],
        backend: DataTableBackend,
    ):
        """Initialize a Dataset with a backend instance.

        Note: This constructor is primarily for internal use.
        For new code, prefer using Dataset.create() or Dataset.get() class methods.

        Args:
            name: The name of the dataset
            model: The Pydantic model class for entries
            project_id: The ID of the parent project
            dataset_id: The ID of this dataset
            datatable_type: Whether this is for "datasets" or "experiments"
            backend: The backend instance to use
        """
        # Store basic properties
        self.name = name
        self.model = model
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.backend_type = getattr(backend, 'backend_type', 'unknown')
        self.datatable_type = datatable_type
        self._entries: t.List[BaseModelType] = []
        self._backend = backend

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


class Dataset(DataTable[BaseModelType]):
    """Dataset class for managing dataset entries.
    
    Inherits all functionality from DataTable. This class represents
    datasets specifically (as opposed to experiments).
    """
    pass
