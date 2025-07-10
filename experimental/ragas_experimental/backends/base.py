"""Base classes for project and dataset backends."""

import typing as t
from abc import ABC, abstractmethod

from pydantic import BaseModel


class BaseBackend(ABC):
    """Abstract base class for datatable backends.

    All datatable storage backends must implement these methods.
    Handles both datasets and experiments.
    """

    @abstractmethod
    def load_dataset(self, name: str) -> t.List[t.Dict[str, t.Any]]:
        """Initialize the backend with dataset information"""
        pass

    @abstractmethod
    def load_experiment(self, name: str) -> t.List[t.Dict[str, t.Any]]:
        pass

    @abstractmethod
    def save_dataset(
        self,
        name: str,
        data: t.List[t.Dict[str, t.Any]],
        data_model: t.Optional[t.Type[BaseModel]],
    ) -> None:
        pass

    @abstractmethod
    def save_experiment(
        self,
        name: str,
        data: t.List[t.Dict[str, t.Any]],
        data_model: t.Optional[t.Type[BaseModel]],
    ) -> None:
        pass

    @abstractmethod
    def list_datasets(self) -> t.List[str]:
        pass

    @abstractmethod
    def list_experiments(self) -> t.List[str]:
        pass
