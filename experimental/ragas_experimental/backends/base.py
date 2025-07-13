"""Base classes for project and dataset backends."""

import typing as t
from abc import ABC, abstractmethod

from pydantic import BaseModel


class BaseBackend(ABC):
    """Abstract base class for dataset and experiment storage backends.

    Backends provide persistent storage for datasets and experiments as lists of dictionaries.
    The system stores datasets and experiments separately but with identical interfaces.

    Implementation Requirements:
    - Handle datasets and experiments with same interface but separate storage
    - Return data as List[Dict[str, Any]] format
    - Raise FileNotFoundError for missing datasets/experiments
    - Support empty datasets (return empty list, not None)
    - Create storage directories/containers as needed

    Directory Structure (for file-based backends):
        storage_root/
        ├── datasets/     # Dataset storage
        └── experiments/  # Experiment storage

    Usage for Implementers:
        class MyBackend(BaseBackend):
            def __init__(self, connection_config):
                self.config = connection_config
                # Initialize your storage connection

            def load_dataset(self, name: str):
                # Load dataset by name, raise FileNotFoundError if missing
                pass

    Usage by End Users:
        # Via string backend registration
        dataset = Dataset("my_data", "my_backend", **backend_config)

        # Via backend instance
        backend = MyBackend(config)
        dataset = Dataset("my_data", backend)
    """

    @abstractmethod
    def load_dataset(self, name: str) -> t.List[t.Dict[str, t.Any]]:
        """Load dataset by name.

        Args:
            name: Dataset identifier (alphanumeric, hyphens, underscores recommended)

        Returns:
            List of dictionaries representing dataset rows. Empty list for empty datasets.

        Raises:
            FileNotFoundError: If dataset doesn't exist

        Implementation Notes:
            - Return empty list [] for empty datasets, never None
            - Each dict represents one data row/item
            - Preserve data types where possible (JSONL) or document limitations (CSV)
        """
        pass

    @abstractmethod
    def load_experiment(self, name: str) -> t.List[t.Dict[str, t.Any]]:
        """Load experiment by name.

        Args:
            name: Experiment identifier (alphanumeric, hyphens, underscores recommended)

        Returns:
            List of dictionaries representing experiment results. Empty list for empty experiments.

        Raises:
            FileNotFoundError: If experiment doesn't exist

        Implementation Notes:
            - Identical interface to load_dataset but separate storage
            - Return empty list [] for empty experiments, never None
        """
        pass

    @abstractmethod
    def save_dataset(
        self,
        name: str,
        data: t.List[t.Dict[str, t.Any]],
        data_model: t.Optional[t.Type[BaseModel]] = None,
    ) -> None:
        """Save dataset with given name.

        Args:
            name: Dataset identifier for storage
            data: List of dictionaries to save
            data_model: Optional Pydantic model for validation context (may be ignored)

        Implementation Notes:
            - Overwrite existing dataset with same name
            - Create storage location if it doesn't exist
            - Handle empty data list gracefully
            - data_model is for context only; data is always pre-validated dicts
        """
        pass

    @abstractmethod
    def save_experiment(
        self,
        name: str,
        data: t.List[t.Dict[str, t.Any]],
        data_model: t.Optional[t.Type[BaseModel]] = None,
    ) -> None:
        """Save experiment with given name.

        Args:
            name: Experiment identifier for storage
            data: List of dictionaries to save
            data_model: Optional Pydantic model for validation context (may be ignored)

        Implementation Notes:
            - Identical interface to save_dataset but separate storage
            - Overwrite existing experiment with same name
        """
        pass

    @abstractmethod
    def list_datasets(self) -> t.List[str]:
        """List all available dataset names.

        Returns:
            Sorted list of dataset names (without file extensions or paths)

        Implementation Notes:
            - Return empty list if no datasets exist
            - Sort alphabetically for consistent ordering
            - Return just the names, not full paths or metadata
        """
        pass

    @abstractmethod
    def list_experiments(self) -> t.List[str]:
        """List all available experiment names.

        Returns:
            Sorted list of experiment names (without file extensions or paths)

        Implementation Notes:
            - Identical interface to list_datasets but for experiments
            - Return empty list if no experiments exist
        """
        pass
