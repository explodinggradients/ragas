"""Base classes for project and dataset backends."""

import typing as t
from abc import ABC, abstractmethod

from ragas_experimental.model.pydantic_model import (
    ExtendedPydanticBaseModel as BaseModel,
)


class DatasetBackend(ABC):
    """Abstract base class for dataset backends.

    All dataset storage backends must implement these methods.
    """

    @abstractmethod
    def initialize(self, dataset: t.Any) -> None:
        """Initialize the backend with dataset information"""
        pass

    @abstractmethod
    def get_column_mapping(self, model: t.Type[BaseModel]) -> t.Dict[str, str]:
        """Get mapping between model fields and backend columns"""
        pass

    @abstractmethod
    def load_entries(self, model_class) -> t.List[t.Any]:
        """Load all entries from storage"""
        pass

    @abstractmethod
    def append_entry(self, entry) -> str:
        """Add a new entry to storage and return its ID"""
        pass

    @abstractmethod
    def update_entry(self, entry) -> bool:
        """Update an existing entry in storage"""
        pass

    @abstractmethod
    def delete_entry(self, entry_id) -> bool:
        """Delete an entry from storage"""
        pass

    @abstractmethod
    def get_entry_by_field(
        self, field_name: str, field_value: t.Any, model_class
    ) -> t.Optional[t.Any]:
        """Get an entry by field value"""
        pass


class ProjectBackend(ABC):
    """Abstract base class for project backends.

    Handles project-level operations like creating/listing datasets and experiments.
    """

    @abstractmethod
    def initialize(self, project_id: str, **kwargs) -> None:
        """Initialize the backend with project information"""
        pass

    @abstractmethod
    def create_dataset(self, name: str, model: t.Type[BaseModel]) -> str:
        """Create a new dataset and return its ID"""
        pass

    @abstractmethod
    def create_experiment(self, name: str, model: t.Type[BaseModel]) -> str:
        """Create a new experiment and return its ID"""
        pass

    @abstractmethod
    def list_datasets(self) -> t.List[t.Dict]:
        """List all datasets in the project"""
        pass

    @abstractmethod
    def list_experiments(self) -> t.List[t.Dict]:
        """List all experiments in the project"""
        pass

    @abstractmethod
    def get_dataset_backend(
        self, dataset_id: str, name: str, model: t.Type[BaseModel]
    ) -> DatasetBackend:
        """Get a DatasetBackend instance for a specific dataset"""
        pass

    @abstractmethod
    def get_experiment_backend(
        self, experiment_id: str, name: str, model: t.Type[BaseModel]
    ) -> DatasetBackend:
        """Get a DatasetBackend instance for a specific experiment"""
        pass

    @abstractmethod
    def get_dataset_by_name(
        self, name: str, model: t.Type[BaseModel]
    ) -> t.Tuple[str, DatasetBackend]:
        """Get dataset ID and backend by name. Returns (dataset_id, backend)"""
        pass

    @abstractmethod
    def get_experiment_by_name(
        self, name: str, model: t.Type[BaseModel]
    ) -> t.Tuple[str, DatasetBackend]:
        """Get experiment ID and backend by name. Returns (experiment_id, backend)"""
        pass
