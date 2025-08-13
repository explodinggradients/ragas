"""In-memory backend for temporary dataset and experiment storage."""

import typing as t
from copy import deepcopy

from pydantic import BaseModel

from .base import BaseBackend


class InMemoryBackend(BaseBackend):
    """Backend that stores datasets and experiments in memory.

    This backend is designed for temporary storage of datasets and experiments
    that don't need persistence. It's particularly useful for:
    - train/test splits that are temporary
    - intermediate datasets during processing
    - testing and development

    Features:
    - No configuration required
    - Preserves all data types exactly (unlike CSV backend)
    - Separate storage for datasets and experiments
    - Instance isolation (multiple instances don't share data)
    - Thread-safe for basic operations

    Usage:
        backend = InMemoryBackend()
        backend.save_dataset("my_dataset", data)
        loaded_data = backend.load_dataset("my_dataset")
    """

    def __init__(self):
        """Initialize the backend with empty storage."""
        self._datasets: t.Dict[str, t.List[t.Dict[str, t.Any]]] = {}
        self._experiments: t.Dict[str, t.List[t.Dict[str, t.Any]]] = {}

    def load_dataset(self, name: str) -> t.List[t.Dict[str, t.Any]]:
        """Load dataset by name.

        Args:
            name: Dataset identifier

        Returns:
            List of dictionaries representing dataset rows. Empty list for empty datasets.

        Raises:
            FileNotFoundError: If dataset doesn't exist
        """
        if name not in self._datasets:
            raise FileNotFoundError(f"Dataset '{name}' not found")

        # Return a deep copy to prevent accidental modification
        return deepcopy(self._datasets[name])

    def load_experiment(self, name: str) -> t.List[t.Dict[str, t.Any]]:
        """Load experiment by name.

        Args:
            name: Experiment identifier

        Returns:
            List of dictionaries representing experiment results. Empty list for empty experiments.

        Raises:
            FileNotFoundError: If experiment doesn't exist
        """
        if name not in self._experiments:
            raise FileNotFoundError(f"Experiment '{name}' not found")

        # Return a deep copy to prevent accidental modification
        return deepcopy(self._experiments[name])

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
            data_model: Optional Pydantic model for validation context (ignored)

        Notes:
            - Overwrites existing dataset with same name
            - Handles empty data list gracefully
            - data_model is ignored (for compatibility with BaseBackend interface)
        """
        # Store a deep copy to prevent accidental modification of original data
        self._datasets[name] = deepcopy(data)

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
            data_model: Optional Pydantic model for validation context (ignored)

        Notes:
            - Overwrites existing experiment with same name
            - Handles empty data list gracefully
            - data_model is ignored (for compatibility with BaseBackend interface)
        """
        # Store a deep copy to prevent accidental modification of original data
        self._experiments[name] = deepcopy(data)

    def list_datasets(self) -> t.List[str]:
        """List all available dataset names.

        Returns:
            Sorted list of dataset names
        """
        return sorted(self._datasets.keys())

    def list_experiments(self) -> t.List[str]:
        """List all available experiment names.

        Returns:
            Sorted list of experiment names
        """
        return sorted(self._experiments.keys())
