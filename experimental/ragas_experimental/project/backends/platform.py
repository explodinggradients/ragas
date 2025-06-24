"""Platform (Ragas API) backend implementation for projects and datasets."""

import asyncio
import typing as t

import ragas_experimental.typing as rt
from ragas_experimental.model.pydantic_model import (
    ExtendedPydanticBaseModel as BaseModel,
)

from ...backends.ragas_api_client import RagasApiClient
from ...utils import async_to_sync
from ..utils import create_nano_id
from .base import DatasetBackend, ProjectBackend


class PlatformDatasetBackend(DatasetBackend):
    """Platform API implementation of DatasetBackend."""

    def __init__(
        self,
        ragas_api_client: RagasApiClient,
        project_id: str,
        dataset_id: str,
        datatable_type: t.Literal["datasets", "experiments"],
    ):
        self.ragas_api_client = ragas_api_client
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.datatable_type = datatable_type
        self.dataset = None

    def initialize(self, dataset):
        """Initialize the backend with the dataset instance."""
        self.dataset = dataset

    def get_column_mapping(self, model):
        """Get mapping between model fields and backend columns."""
        if self.datatable_type == "datasets":
            sync_func = async_to_sync(self.ragas_api_client.list_dataset_columns)
            columns = sync_func(project_id=self.project_id, dataset_id=self.dataset_id)
        else:  # experiments
            sync_func = async_to_sync(self.ragas_api_client.list_experiment_columns)
            columns = sync_func(
                project_id=self.project_id, experiment_id=self.dataset_id
            )

        column_id_map = {column["name"]: column["id"] for column in columns["items"]}

        # Update the model's column mapping with the values from the API
        column_mapping = {}
        for field_name in model.__annotations__:
            if field_name in column_id_map:
                column_mapping[field_name] = column_id_map[field_name]

        return column_mapping

    def load_entries(self, model_class) -> t.List[t.Any]:
        """Load all entries from the API."""
        # Get all rows
        if self.datatable_type == "datasets":
            sync_func = async_to_sync(self.ragas_api_client.list_dataset_rows)
            response = sync_func(project_id=self.project_id, dataset_id=self.dataset_id)
        else:  # experiments
            sync_func = async_to_sync(self.ragas_api_client.list_experiment_rows)
            response = sync_func(
                project_id=self.project_id, experiment_id=self.dataset_id
            )

        # Get column mapping (ID -> name)
        column_map = {v: k for k, v in model_class.__column_mapping__.items()}

        # Process rows
        entries = []
        for row in response.get("items", []):
            model_data = {}
            row_id = row.get("id")

            # Convert from API data format to model fields
            for col_id, value in row.get("data", {}).items():
                if col_id in column_map:
                    field_name = column_map[col_id]
                    model_data[field_name] = value

            # Create model instance
            entry = model_class(**model_data)

            # Store row ID for future operations
            entry._row_id = row_id

            entries.append(entry)

        return entries

    def append_entry(self, entry) -> str:
        """Add a new entry to the API and return its ID."""
        # Get column mapping
        column_id_map = entry.__class__.__column_mapping__

        # Create row data
        row_dict_converted = rt.ModelConverter.instance_to_row(entry)
        row_id = create_nano_id()
        row_data = {}

        for column in row_dict_converted["data"]:
            if column["column_id"] in column_id_map:
                row_data[column_id_map[column["column_id"]]] = column["data"]

        # Create row in API
        if self.datatable_type == "datasets":
            sync_func = async_to_sync(self.ragas_api_client.create_dataset_row)
            response = sync_func(
                project_id=self.project_id,
                dataset_id=self.dataset_id,
                id=row_id,
                data=row_data,
            )
        else:  # experiments
            sync_func = async_to_sync(self.ragas_api_client.create_experiment_row)
            response = sync_func(
                project_id=self.project_id,
                experiment_id=self.dataset_id,
                id=row_id,
                data=row_data,
            )

        # Return the row ID
        return response["id"]

    def update_entry(self, entry) -> bool:
        """Update an existing entry in the API."""
        # Get the row ID
        row_id = None
        if hasattr(entry, "_row_id") and entry._row_id:
            row_id = entry._row_id
        else:
            raise ValueError("Cannot update: entry has no row ID")

        # Get column mapping and prepare data
        column_id_map = entry.__class__.__column_mapping__
        row_dict = rt.ModelConverter.instance_to_row(entry)["data"]
        row_data = {}

        for column in row_dict:
            if column["column_id"] in column_id_map:
                row_data[column_id_map[column["column_id"]]] = column["data"]

        # Update in API
        if self.datatable_type == "datasets":
            sync_func = async_to_sync(self.ragas_api_client.update_dataset_row)
            response = sync_func(
                project_id=self.project_id,
                dataset_id=self.dataset_id,
                row_id=row_id,
                data=row_data,
            )
        else:  # experiments
            sync_func = async_to_sync(self.ragas_api_client.update_experiment_row)
            response = sync_func(
                project_id=self.project_id,
                experiment_id=self.dataset_id,
                row_id=row_id,
                data=row_data,
            )

        return response

    def delete_entry(self, entry_id) -> bool:
        """Delete an entry from the API."""
        # Delete the row
        if self.datatable_type == "datasets":
            sync_func = async_to_sync(self.ragas_api_client.delete_dataset_row)
            response = sync_func(
                project_id=self.project_id, dataset_id=self.dataset_id, row_id=entry_id
            )
        else:  # experiments
            sync_func = async_to_sync(self.ragas_api_client.delete_experiment_row)
            response = sync_func(
                project_id=self.project_id,
                experiment_id=self.dataset_id,
                row_id=entry_id,
            )

        return response

    def get_entry_by_field(
        self, field_name, field_value, model_class
    ) -> t.Optional[t.Any]:
        """Get an entry by field value."""
        # We don't have direct filtering in the API, so load all and filter
        entries = self.load_entries(model_class)

        # Search for matching entry
        for entry in entries:
            if hasattr(entry, field_name) and getattr(entry, field_name) == field_value:
                return entry

        return None


async def create_dataset_columns(
    project_id, dataset_id, columns, create_dataset_column_func
):
    """Helper function to create dataset columns."""
    tasks = []
    for column in columns:
        tasks.append(
            create_dataset_column_func(
                project_id=project_id,
                dataset_id=dataset_id,
                id=create_nano_id(),
                name=column["name"],
                type=column["type"],
                settings=column["settings"],
            )
        )
    return await asyncio.gather(*tasks)


async def create_experiment_columns(
    project_id, experiment_id, columns, create_experiment_column_func
):
    """Helper function to create experiment columns."""
    tasks = []
    for column in columns:
        tasks.append(
            create_experiment_column_func(
                project_id=project_id,
                experiment_id=experiment_id,
                id=create_nano_id(),
                name=column["name"],
                type=column["type"],
                settings=column["settings"],
            )
        )
    return await asyncio.gather(*tasks)


class PlatformProjectBackend(ProjectBackend):
    """Platform API implementation of ProjectBackend."""

    def __init__(self, ragas_api_client: RagasApiClient):
        self.ragas_api_client = ragas_api_client
        self.project_id: t.Optional[str] = None

    def initialize(self, project_id: str, **kwargs):
        """Initialize the backend with project information."""
        self.project_id = project_id

    def create_dataset(self, name: str, model: t.Type[BaseModel]) -> str:
        """Create a new dataset and return its ID."""
        # Create the dataset
        sync_version = async_to_sync(self.ragas_api_client.create_dataset)
        dataset_info = sync_version(
            project_id=self.project_id,
            name=name,
        )

        # Create the columns for the dataset
        column_types = rt.ModelConverter.model_to_columns(model)
        sync_create_columns = async_to_sync(create_dataset_columns)
        sync_create_columns(
            project_id=self.project_id,
            dataset_id=dataset_info["id"],
            columns=column_types,
            create_dataset_column_func=self.ragas_api_client.create_dataset_column,
        )

        return dataset_info["id"]

    def create_experiment(self, name: str, model: t.Type[BaseModel]) -> str:
        """Create a new experiment and return its ID."""
        # Create the experiment in the API
        sync_version = async_to_sync(self.ragas_api_client.create_experiment)
        experiment_info = sync_version(
            project_id=self.project_id,
            name=name,
        )

        # Create the columns for the experiment
        column_types = rt.ModelConverter.model_to_columns(model)
        sync_version = async_to_sync(create_experiment_columns)
        sync_version(
            project_id=self.project_id,
            experiment_id=experiment_info["id"],
            columns=column_types,
            create_experiment_column_func=self.ragas_api_client.create_experiment_column,
        )

        return experiment_info["id"]

    def list_datasets(self) -> t.List[t.Dict]:
        """List all datasets in the project."""
        sync_version = async_to_sync(self.ragas_api_client.list_datasets)
        datasets = sync_version(project_id=self.project_id)
        return datasets.get("items", [])

    def list_experiments(self) -> t.List[t.Dict]:
        """List all experiments in the project."""
        sync_version = async_to_sync(self.ragas_api_client.list_experiments)
        experiments = sync_version(project_id=self.project_id)
        return experiments.get("items", [])

    def get_dataset_backend(
        self, dataset_id: str, name: str, model: t.Type[BaseModel]
    ) -> DatasetBackend:
        """Get a DatasetBackend instance for a specific dataset."""
        if self.project_id is None:
            raise ValueError(
                "Backend must be initialized before creating dataset backend"
            )
        return PlatformDatasetBackend(
            ragas_api_client=self.ragas_api_client,
            project_id=self.project_id,
            dataset_id=dataset_id,
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
        return PlatformDatasetBackend(
            ragas_api_client=self.ragas_api_client,
            project_id=self.project_id,
            dataset_id=experiment_id,
            datatable_type="experiments",
        )

    def get_dataset_by_name(
        self, name: str, model: t.Type[BaseModel]
    ) -> t.Tuple[str, DatasetBackend]:
        """Get dataset ID and backend by name."""
        # Search for dataset with given name
        sync_version = async_to_sync(self.ragas_api_client.get_dataset_by_name)
        dataset_info = sync_version(project_id=self.project_id, dataset_name=name)

        backend = self.get_dataset_backend(dataset_info["id"], name, model)
        return dataset_info["id"], backend

    def get_experiment_by_name(
        self, name: str, model: t.Type[BaseModel]
    ) -> t.Tuple[str, DatasetBackend]:
        """Get experiment ID and backend by name."""
        # Search for experiment with given name
        sync_version = async_to_sync(self.ragas_api_client.get_experiment_by_name)
        experiment_info = sync_version(project_id=self.project_id, experiment_name=name)

        backend = self.get_experiment_backend(experiment_info["id"], name, model)
        return experiment_info["id"], backend
