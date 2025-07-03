"""Local CSV backend implementation for projects and datasets."""

import csv
import os
import typing as t
import uuid

from ragas_experimental.model.pydantic_model import (
    ExtendedPydanticBaseModel as BaseModel,
)

from ..utils import create_nano_id
from .base import DatasetBackend, ProjectBackend


class LocalCSVDatasetBackend(DatasetBackend):
    """Local CSV implementation of DatasetBackend."""

    def __init__(
        self,
        local_root_dir: str,
        project_id: str,
        dataset_id: str,
        dataset_name: str,
        datatable_type: t.Literal["datasets", "experiments"],
    ):
        self.local_root_dir = local_root_dir
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.datatable_type = datatable_type
        self.dataset = None

    def initialize(self, dataset):
        """Initialize the backend with the dataset instance."""
        self.dataset = dataset
        self._ensure_csv_exists()

    def _ensure_csv_exists(self):
        """Create the CSV file if it doesn't exist."""
        csv_path = self._get_csv_path()

        # Create directories if needed
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        # Create file with headers if it doesn't exist
        if not os.path.exists(csv_path):
            # Include _row_id in the headers
            if self.dataset is None:
                raise ValueError(
                    "Dataset must be initialized before creating CSV headers"
                )
            field_names = ["_row_id"] + list(self.dataset.model.__annotations__.keys())

            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(field_names)

    def _get_csv_path(self):
        """Get the path to the CSV file."""
        return os.path.join(
            self.local_root_dir,
            self.project_id,
            self.datatable_type,
            f"{self.dataset_name}.csv",
        )

    def get_column_mapping(self, model) -> t.Dict:
        """Get mapping between model fields and CSV columns."""
        return model.model_fields

    def load_entries(self, model_class):
        """Load all entries from the CSV file."""
        csv_path = self._get_csv_path()

        if not os.path.exists(csv_path):
            return []

        entries = []

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    # Extract row_id and remove from model data
                    row_id = row.get("_row_id", str(uuid.uuid4()))

                    # Create a copy without _row_id for model instantiation
                    model_data = {k: v for k, v in row.items() if k != "_row_id"}

                    # Convert types as needed
                    typed_row = {}
                    for field, value in model_data.items():
                        if field in model_class.model_fields:
                            field_type = model_class.model_fields[field].annotation

                            # Handle basic type conversions
                            if field_type is int:
                                typed_row[field] = int(value) if value else 0
                            elif field_type is float:
                                typed_row[field] = float(value) if value else 0.0
                            elif field_type is bool:
                                typed_row[field] = value.lower() in (
                                    "true",
                                    "t",
                                    "yes",
                                    "y",
                                    "1",
                                )
                            else:
                                typed_row[field] = value

                    # Create model instance
                    entry = model_class(**typed_row)

                    # Set the row ID from CSV
                    entry._row_id = row_id

                    entries.append(entry)
                except Exception as e:
                    print(f"Error loading row from CSV: {e}")

        return entries

    def append_entry(self, entry) -> str:
        """Add a new entry to the CSV file and return a generated ID."""
        csv_path = self._get_csv_path()

        # Read existing rows to avoid overwriting
        existing_rows = []
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)

        # Generate a row ID if needed
        row_id = getattr(entry, "_row_id", None) or str(uuid.uuid4())

        # Get field names including row_id
        field_names = ["_row_id"] + list(entry.model_fields.keys())

        # Convert entry to dict
        entry_dict = entry.model_dump()

        # Add row_id to the dict
        entry_dict["_row_id"] = row_id

        # Write all rows back with the new entry
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()

            # Write existing rows
            for row in existing_rows:
                writer.writerow(row)

            # Write new row
            writer.writerow(entry_dict)

        # Return the row ID
        return row_id

    def update_entry(self, entry) -> bool:
        """Update an existing entry in the CSV file."""
        # Create a copy of entries to modify
        if self.dataset is None:
            raise ValueError("Dataset must be initialized")
        entries_to_save = list(self.dataset._entries)  # Make a copy

        # Find the entry to update
        updated = False
        for i, e in enumerate(entries_to_save):
            if (
                hasattr(e, "_row_id")
                and hasattr(entry, "_row_id")
                and e._row_id == entry._row_id
            ):
                # Update the entry in our copy
                entries_to_save[i] = entry
                updated = True
                break

        # If entry wasn't found, just append it
        if not updated and entries_to_save:
            entries_to_save.append(entry)

        # Write all entries back to CSV
        self._write_entries_to_csv(entries_to_save)

        return True

    def delete_entry(self, entry_id) -> bool:
        """Delete an entry from the CSV file."""
        # Create a copy of entries to modify, excluding the one to delete
        if self.dataset is None:
            raise ValueError("Dataset must be initialized")
        entries_to_save = []
        for e in self.dataset._entries:
            if not (hasattr(e, "_row_id") and e._row_id == entry_id):
                entries_to_save.append(e)

        # Write all entries back to CSV
        self._write_entries_to_csv(entries_to_save)

        return True

    def _write_entries_to_csv(self, entries):
        """Write all entries to the CSV file."""
        csv_path = self._get_csv_path()

        if not entries:
            # If no entries, just create an empty CSV with headers
            if self.dataset is None:
                raise ValueError("Dataset must be initialized")
            field_names = ["_row_id"] + list(self.dataset.model.model_fields.keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=field_names)
                writer.writeheader()
            return

        # Get field names including _row_id
        field_names = ["_row_id"] + list(entries[0].__class__.model_fields.keys())

        # Write all entries
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()

            for entry in entries:
                # Create a dict with model data + row_id
                entry_dict = entry.model_dump()
                entry_dict["_row_id"] = getattr(entry, "_row_id", str(uuid.uuid4()))

                writer.writerow(entry_dict)

    def get_entry_by_field(
        self, field_name, field_value, model_class
    ) -> t.Optional[t.Any]:
        """Get an entry by field value."""
        entries = self.load_entries(model_class)

        for entry in entries:
            if hasattr(entry, field_name) and getattr(entry, field_name) == field_value:
                return entry

        return None


class LocalCSVProjectBackend(ProjectBackend):
    """Local CSV implementation of ProjectBackend."""

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.project_id: t.Optional[str] = None

    def initialize(self, project_id: str, **kwargs):
        """Initialize the backend with project information."""
        self.project_id = project_id
        self._project_dir = os.path.join(self.root_dir, project_id)
        self._create_project_structure()

    def _create_project_structure(self):
        """Create the local directory structure for the project."""
        os.makedirs(self._project_dir, exist_ok=True)
        # Create datasets directory
        os.makedirs(os.path.join(self._project_dir, "datasets"), exist_ok=True)
        # Create experiments directory
        os.makedirs(os.path.join(self._project_dir, "experiments"), exist_ok=True)

    def create_dataset(self, name: str, model: t.Type[BaseModel]) -> str:
        """Create a new dataset and return its ID."""
        dataset_id = create_nano_id()
        return dataset_id

    def create_experiment(self, name: str, model: t.Type[BaseModel]) -> str:
        """Create a new experiment and return its ID."""
        experiment_id = create_nano_id()
        return experiment_id

    def list_datasets(self) -> t.List[t.Dict]:
        """List all datasets in the project."""
        datasets_dir = os.path.join(self._project_dir, "datasets")
        if not os.path.exists(datasets_dir):
            return []

        datasets = []
        for filename in os.listdir(datasets_dir):
            if filename.endswith(".csv"):
                name = os.path.splitext(filename)[0]
                datasets.append(
                    {
                        "id": create_nano_id(),  # Generate ID for consistency
                        "name": name,
                    }
                )
        return datasets

    def list_experiments(self) -> t.List[t.Dict]:
        """List all experiments in the project."""
        experiments_dir = os.path.join(self._project_dir, "experiments")
        if not os.path.exists(experiments_dir):
            return []

        experiments = []
        for filename in os.listdir(experiments_dir):
            if filename.endswith(".csv"):
                name = os.path.splitext(filename)[0]
                experiments.append(
                    {
                        "id": create_nano_id(),  # Generate ID for consistency
                        "name": name,
                    }
                )
        return experiments

    def get_dataset_backend(
        self, dataset_id: str, name: str, model: t.Type[BaseModel]
    ) -> DatasetBackend:
        """Get a DatasetBackend instance for a specific dataset."""
        if self.project_id is None:
            raise ValueError(
                "Backend must be initialized before creating dataset backend"
            )
        return LocalCSVDatasetBackend(
            local_root_dir=self.root_dir,
            project_id=self.project_id,
            dataset_id=dataset_id,
            dataset_name=name,
            datatable_type="datasets",
        )

    def get_experiment_backend(
        self, experiment_id: str, name: str, model: t.Type[BaseModel]
    ) -> DatasetBackend:
        """Get a DatasetBackend instance for a specific experiment."""
        if self.project_id is None:
            raise ValueError(
                "Backend must be initialized before creating experiment backend"
            )
        return LocalCSVDatasetBackend(
            local_root_dir=self.root_dir,
            project_id=self.project_id,
            dataset_id=experiment_id,
            dataset_name=name,
            datatable_type="experiments",
        )

    def get_dataset_by_name(
        self, name: str, model: t.Type[BaseModel]
    ) -> t.Tuple[str, DatasetBackend]:
        """Get dataset ID and backend by name."""
        # Check if the dataset file exists
        dataset_path = os.path.join(self._project_dir, "datasets", f"{name}.csv")
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset '{name}' does not exist in path {dataset_path}")

        # Create dataset instance with a random ID
        dataset_id = create_nano_id()
        backend = self.get_dataset_backend(dataset_id, name, model)

        return dataset_id, backend

    def get_experiment_by_name(
        self, name: str, model: t.Type[BaseModel]
    ) -> t.Tuple[str, DatasetBackend]:
        """Get experiment ID and backend by name."""
        # Check if the experiment file exists
        experiment_path = os.path.join(self._project_dir, "experiments", f"{name}.csv")
        if not os.path.exists(experiment_path):
            raise ValueError(f"Experiment '{name}' does not exist")

        # Create experiment instance with a random ID
        experiment_id = create_nano_id()
        backend = self.get_experiment_backend(experiment_id, name, model)

        return experiment_id, backend
