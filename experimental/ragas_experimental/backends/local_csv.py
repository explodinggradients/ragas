"""Local CSV backend implementation for projects and datasets."""

import csv
import typing as t
from pathlib import Path

from pydantic import BaseModel

from .base import BaseBackend


class LocalCSVBackend(BaseBackend):
    """Local CSV implementation of DataTableBackend."""

    def __init__(
        self,
        root_dir: str,
    ):
        self.root_dir = Path(root_dir)

    def _get_data_dir(self, data_type: str) -> Path:
        """Get the directory path for datasets or experiments."""
        return self.root_dir / data_type

    def _get_file_path(self, data_type: str, name: str) -> Path:
        """Get the full file path for a dataset or experiment."""
        return self._get_data_dir(data_type) / f"{name}.csv"

    def _load(self, data_type: str, name: str) -> t.List[t.Dict[str, t.Any]]:
        """Load data from CSV file, raising FileNotFoundError if file doesn't exist."""
        file_path = self._get_file_path(data_type, name)

        if not file_path.exists():
            raise FileNotFoundError(
                f"No {data_type[:-1]} named '{name}' found at {file_path}"
            )

        with open(file_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def _save(
        self,
        data_type: str,
        name: str,
        data: t.List[t.Dict[str, t.Any]],
        data_model: t.Optional[t.Type[BaseModel]],
    ) -> None:
        """Save data to CSV file, creating directory if needed."""
        file_path = self._get_file_path(data_type, name)

        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Handle empty data
        if not data:
            # Create empty CSV file
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                pass
            return

        # Write data to CSV
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = data[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    def _list(self, data_type: str) -> t.List[str]:
        """List all available datasets or experiments."""
        data_dir = self._get_data_dir(data_type)

        if not data_dir.exists():
            return []

        # Get all .csv files and return names without extension
        csv_files = [f.stem for f in data_dir.glob("*.csv")]
        return sorted(csv_files)

    # Public interface methods (required by BaseBackend)
    def load_dataset(self, name: str) -> t.List[t.Dict[str, t.Any]]:
        """Load a dataset from CSV file."""
        return self._load("datasets", name)

    def load_experiment(self, name: str) -> t.List[t.Dict[str, t.Any]]:
        """Load an experiment from CSV file."""
        return self._load("experiments", name)

    def save_dataset(
        self,
        name: str,
        data: t.List[t.Dict[str, t.Any]],
        data_model: t.Optional[t.Type[BaseModel]] = None,
    ) -> None:
        """Save a dataset to CSV file."""
        self._save("datasets", name, data, data_model)

    def save_experiment(
        self,
        name: str,
        data: t.List[t.Dict[str, t.Any]],
        data_model: t.Optional[t.Type[BaseModel]] = None,
    ) -> None:
        """Save an experiment to CSV file."""
        self._save("experiments", name, data, data_model)

    def list_datasets(self) -> t.List[str]:
        """List all available datasets."""
        return self._list("datasets")

    def list_experiments(self) -> t.List[str]:
        """List all available experiments."""
        return self._list("experiments")

    def __repr__(self) -> str:
        return f"LocalCSVBackend(root_dir='{self.root_dir}')"

    __str__ = __repr__
