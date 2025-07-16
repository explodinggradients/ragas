"""A python list like object that contains your evaluation data."""

__all__ = [
    "DataTable",
    "Dataset",
]

import typing as t

from pydantic import BaseModel

from .backends import BaseBackend, get_registry

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

    @t.overload
    def __init__(
        self,
        name: str,
        backend: str,
        data_model: t.Type[T],
        data: t.Optional[t.List[T]] = None,
        **kwargs,
    ) -> None: ...

    @t.overload
    def __init__(
        self,
        name: str,
        backend: str,
        data_model: None = None,
        data: t.Optional[t.List[t.Dict[str, t.Any]]] = None,
        **kwargs,
    ) -> None: ...
    def __init__(
        self,
        name: str,
        backend: t.Union[BaseBackend, str],
        data_model: t.Optional[t.Type[T]] = None,
        data: t.Optional[t.List[t.Any]] = None,
        **kwargs,
    ):
        """Initialize a Dataset with a backend.

        Args:
            name: The name of the dataset
            backend: Either a BaseBackend instance or backend name string (e.g., "local/csv")
            data_model: Optional Pydantic model class for entries
            data: Optional initial data list
            **kwargs: Additional arguments passed to backend constructor (when using string backend)

        Examples:
            # Using string backend name
            dataset = Dataset("my_data", "local/csv", root_dir="./data")

            # Using backend instance (existing behavior)
            backend = LocalCSVBackend(root_dir="./data")
            dataset = Dataset("my_data", backend)
        """
        # Store basic properties
        self.name = name
        self.data_model = data_model
        # Resolve backend if string
        self.backend = self._resolve_backend(backend, **kwargs)
        self._data: t.List[t.Union[t.Dict, T]] = data or []

    @staticmethod
    def _resolve_backend(backend: t.Union[BaseBackend, str], **kwargs) -> BaseBackend:
        """Resolve backend from string or return existing BaseBackend instance.

        Args:
            backend: Either a BaseBackend instance or backend name string (e.g., "local/csv")
            **kwargs: Additional arguments passed to backend constructor (when using string backend)

        Returns:
            BaseBackend instance

        Raises:
            ValueError: If backend string is not found in registry
            TypeError: If backend is wrong type or constructor fails
            RuntimeError: If backend initialization fails
        """
        if isinstance(backend, str):
            registry = get_registry()
            try:
                backend_class = registry[backend]
            except KeyError:
                available = list(registry.keys())
                raise ValueError(
                    f"Backend '{backend}' not found. "
                    f"Available backends: {available}. "
                    f"Install a backend plugin or check the name."
                )

            try:
                return backend_class(**kwargs)
            except TypeError as e:
                raise TypeError(
                    f"Failed to create {backend} backend: {e}. "
                    f"Check required arguments for {backend_class.__name__}."
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize {backend} backend: {e}")

        # Validate backend type
        if not isinstance(backend, BaseBackend):
            raise TypeError(
                f"Backend must be BaseBackend instance or string, got {type(backend)}"
            )

        return backend

    @classmethod
    def load(
        cls: t.Type[Self],
        name: str,
        backend: t.Union[BaseBackend, str],
        data_model: t.Optional[t.Type[T]] = None,
        **kwargs,
    ) -> Self:
        """Load dataset with optional validation.

        Args:
            name: Name of the dataset to load
            backend: Either a BaseBackend instance or backend name string (e.g., "local/csv")
            data_model: Optional Pydantic model for validation
            **kwargs: Additional arguments passed to backend constructor (when using string backend)

        Returns:
            Dataset instance with loaded data

        Examples:
            # Using string backend name
            dataset = Dataset.load("my_data", "local/csv", root_dir="./data")

            # Using backend instance (existing behavior)
            backend = LocalCSVBackend(root_dir="./data")
            dataset = Dataset.load("my_data", backend)
        """
        # Resolve backend if string
        backend = cls._resolve_backend(backend, **kwargs)

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
                raise TypeError(f"Item must be {self.data_model.__name__} or dict")
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

        # Create new dataset instances with proper initialization
        # Handle type-safe constructor calls based on data_model presence
        if self.data_model is not None:
            # Validated dataset case - data should be List[T]
            train_data = t.cast(t.List[T], self._data[:split_index])
            test_data = t.cast(t.List[T], self._data[split_index:])

            train_dataset = type(self)(
                name=f"{self.name}_train",
                backend=self.backend,
                data_model=self.data_model,
                data=train_data,
            )

            test_dataset = type(self)(
                name=f"{self.name}_test",
                backend=self.backend,
                data_model=self.data_model,
                data=test_data,
            )
        else:
            # Unvalidated dataset case - data should be List[Dict]
            train_data = t.cast(t.List[t.Dict[str, t.Any]], self._data[:split_index])
            test_data = t.cast(t.List[t.Dict[str, t.Any]], self._data[split_index:])

            train_dataset = type(self)(
                name=f"{self.name}_train",
                backend=self.backend,
                data_model=None,
                data=train_data,
            )

            test_dataset = type(self)(
                name=f"{self.name}_test",
                backend=self.backend,
                data_model=None,
                data=test_data,
            )

        return train_dataset, test_dataset
