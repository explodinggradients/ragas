"""A python list like object that contains your evaluation data."""

__all__ = [
    "DataTable",
    "Dataset",
]

import typing as t

from pydantic import BaseModel

from .backends import BaseBackend

# For backwards compatibility, use typing_extensions for older Python versions
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

T = t.TypeVar("T", bound=BaseModel)
DataTableType = t.TypeVar("DataTableType", bound="DataTable")


class DataTable(t.Generic[T]):
    """A list-like interface for managing datatable entries with backend save and load.

    This class behaves like a Python list while synchronizing operations with the
    chosen backend (Ragas API or local filesystem). Base class for Dataset and Experiment.
    """

    DATATABLE_TYPE: t.Literal["Dataset", "Experiment"]

    @t.overload
    def __init__(
        self,
        name: str,
        backend: BaseBackend,
        data_model: t.Type[T],
        data: t.Optional[t.List[T]] = None,
    ) -> None: ...

    @t.overload
    def __init__(
        self,
        name: str,
        backend: BaseBackend,
        data_model: None = None,
        data: t.Optional[t.List[t.Dict[str, t.Any]]] = None,
    ) -> None: ...
    def __init__(
        self,
        name: str,
        backend: BaseBackend,
        data_model: t.Optional[t.Type[T]] = None,
        data: t.Optional[t.List[t.Any]] = None,
    ):
        """Initialize a Dataset with a backend instance.

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
        self.backend = backend
        self.data_model = data_model
        self._data: t.List[t.Union[t.Dict, T]] = data or []

    @classmethod
    def load(
        cls: t.Type[Self],
        name: str,
        backend: BaseBackend,
        data_model: t.Optional[t.Type[T]] = None,
    ) -> Self:
        """Load dataset with optional validation"""
        # Backend always returns dicts
        # Use the correct backend method based on the class type
        if hasattr(cls, "DATATABLE_TYPE") and cls.DATATABLE_TYPE == "Experiment":
            dict_data = backend.load_experiment(name)
        else:
            dict_data = backend.load_dataset(name)

        if data_model:
            # Validated mode - convert dicts to Pydantic models
            validated_data = [data_model(**d) for d in dict_data]
            return cls(name, backend, data_model, validated_data)
        else:
            # Unvalidated mode - keep as dicts but wrapped in Dataset API
            return cls(name, backend, None, dict_data)

    def save(self) -> None:
        """Save dataset - converts to dicts if needed"""
        dict_data: t.List[t.Dict[str, t.Any]] = []

        for item in self._data:
            if isinstance(item, BaseModel):
                dict_data.append(item.model_dump())
            elif isinstance(item, dict):
                dict_data.append(item)
            else:
                raise TypeError(f"Unexpected type in dataset: {type(item)}")

        # Backend only sees dicts
        # Use the correct backend method based on the class type
        if hasattr(self, "DATATABLE_TYPE") and self.DATATABLE_TYPE == "Experiment":
            self.backend.save_experiment(
                self.name, dict_data, data_model=self.data_model
            )
        else:
            self.backend.save_dataset(self.name, dict_data, data_model=self.data_model)

    def validate_with(self, data_model: t.Type[T]) -> Self:
        """Apply validation to an unvalidated dataset"""
        if self.data_model is not None:
            raise ValueError(
                f"Dataset already validated with {self.data_model.__name__}"
            )

        # Ensure all items are dicts before validating
        dict_data: t.List[t.Dict[str, t.Any]] = []
        for item in self._data:
            if isinstance(item, dict):
                dict_data.append(item)
            else:
                raise TypeError("Can only validate datasets containing dictionaries")

        # Validate each row
        validated_data = [data_model(**d) for d in dict_data]

        # Return new validated dataset with same type as self
        return type(self)(
            name=self.name,
            backend=self.backend,
            data_model=data_model,
            data=validated_data,
        )

    def append(self, item: t.Union[t.Dict, BaseModel]) -> None:
        """Add item to dataset with validation if model exists"""
        if self.data_model is not None:
            # Ensure item matches our model
            if isinstance(item, dict):
                validated_item = self.data_model(**item)
                self._data.append(validated_item)
            elif isinstance(item, BaseModel):  # Changed this line
                # Additional check to ensure it's the right model type
                if type(item) is self.data_model:
                    self._data.append(item)
                else:
                    raise TypeError(f"Item must be {self.data_model.__name__} or dict")
            else:
                raise TypeError(f"Item must be {self.data_model.__name__} or dict")  # type: ignore[unreachable]
        else:
            # No model - only accept dicts
            if isinstance(item, dict):
                self._data.append(item)
            else:
                raise TypeError("Dataset without model can only accept dicts")

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]

    def __iter__(self):
        return iter(self._data)

    def __str__(self):
        data_model_str = (
            f"model={self.data_model.__name__}, " if self.data_model else ""
        )

        return f"{self.DATATABLE_TYPE}(name={self.name}, {data_model_str} len={len(self._data)})"

    __repr__ = __str__


class Dataset(DataTable[T]):
    """Dataset class for managing dataset entries.

    Inherits all functionality from DataTable. This class represents
    datasets specifically (as opposed to experiments).
    """

    DATATABLE_TYPE = "Dataset"

    def train_test_split(
        self, test_size: float = 0.2, random_state: t.Optional[int] = None
    ) -> t.Tuple["Dataset[T]", "Dataset[T]"]:
        """Split the dataset into training and testing sets.

        Args:
            test_size: Proportion of the dataset to include in the test split (default: 0.2)
            random_state: Random seed for reproducibility (default: None)
        Returns:
            A tuple of two Datasets: (train_dataset, test_dataset)
        """
        if not self._data:
            self.load(self.name, self.backend, self.data_model)

        # Shuffle entries if random_state is set
        if random_state is not None:
            import random

            random.seed(random_state)
            random.shuffle(self._data)

        # Calculate split index
        split_index = int(len(self._data) * (1 - test_size))

        # Create new dataset instances without full initialization
        train_dataset = object.__new__(type(self))
        test_dataset = object.__new__(type(self))

        # Copy essential attributes
        for dataset in [train_dataset, test_dataset]:
            dataset.data_model = self.data_model
            dataset.backend = self.backend

        # Set specific attributes for each dataset
        train_dataset.name = f"{self.name}_train"

        test_dataset.name = f"{self.name}_test"

        # Assign entries to the new datasets
        train_dataset._data = self._data[:split_index]
        test_dataset._data = self._data[split_index:]

        return train_dataset, test_dataset
