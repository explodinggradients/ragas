"""Python client to api.ragas.io"""

__all__ = [
    "DEFAULT_SETTINGS",
    "RagasApiClient",
    "create_nano_id",
    "Column",
    "RowCell",
    "Row",
]

import asyncio
import string
import typing as t
import uuid

import httpx
from fastcore.utils import patch
from pydantic import BaseModel, Field

from ragas_experimental.exceptions import (
    DatasetNotFoundError,
    DuplicateDatasetError,
    DuplicateExperimentError,
    DuplicateProjectError,
    ExperimentNotFoundError,
    ProjectNotFoundError,
)


class RagasApiClient:
    """Client for the Ragas Relay API."""

    def __init__(self, base_url: str, app_token: t.Optional[str] = None):
        """Initialize the Ragas API client.

        Args:
            base_url: Base URL for the API (e.g., "http://localhost:8087")
            app_token: API token for authentication
        """
        if not app_token:
            raise ValueError("app_token must be provided")

        self.base_url = f"{base_url.rstrip('/')}/api/v1"
        self.app_token = app_token

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: t.Optional[t.Dict] = None,
        json_data: t.Optional[t.Dict] = None,
    ) -> t.Dict:
        """Make a request to the API.

        Args:
            method: HTTP method (GET, POST, PATCH, DELETE)
            endpoint: API endpoint path
            params: Query parameters
            json_data: JSON request body

        Returns:
            The response data from the API
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {"X-App-Token": self.app_token}

        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=method, url=url, params=params, json=json_data, headers=headers
            )

            data = response.json()

            if response.status_code >= 400 or data.get("status") == "error":
                error_msg = data.get("message", "Unknown error")
                raise Exception(f"API Error ({response.status_code}): {error_msg}")

            return data.get("data")

    # ---- Resource Handlers ----
    async def _create_resource(self, path, data):
        """Generic resource creation."""
        return await self._request("POST", path, json_data=data)

    async def _list_resources(self, path, **params):
        """Generic resource listing."""
        return await self._request("GET", path, params=params)

    async def _get_resource(self, path):
        """Generic resource retrieval."""
        return await self._request("GET", path)

    async def _update_resource(self, path, data):
        """Generic resource update."""
        return await self._request("PATCH", path, json_data=data)

    async def _delete_resource(self, path):
        """Generic resource deletion."""
        return await self._request("DELETE", path)


@patch
async def _get_resource_by_name(
    self: RagasApiClient,
    list_method: t.Callable,
    get_method: t.Callable,
    resource_name: str,
    name_field: str,
    not_found_error: t.Type[Exception],
    duplicate_error: t.Type[Exception],
    resource_type_name: str,
    **list_method_kwargs,
) -> t.Dict:
    """Generic method to get a resource by name.

    Args:
        list_method: Method to list resources
        get_method: Method to get a specific resource
        resource_name: Name to search for
        name_field: Field name that contains the resource name
        not_found_error: Exception to raise when resource is not found
        duplicate_error: Exception to raise when multiple resources are found
        resource_type_name: Human-readable name of the resource type
        **list_method_kwargs: Additional arguments to pass to list_method

    Returns:
        The resource information dictionary

    Raises:
        Exception: If resource is not found or multiple resources are found
    """
    # Initial pagination parameters
    limit = 50  # Number of items per page
    offset = 0  # Starting position
    matching_resources = []

    while True:
        # Get a page of resources
        response = await list_method(limit=limit, offset=offset, **list_method_kwargs)

        items = response.get("items", [])

        # If no items returned, we've reached the end
        if not items:
            break

        # Collect all resources with the matching name in this page
        for resource in items:
            if resource.get(name_field) == resource_name:
                matching_resources.append(resource)

        # Update offset for the next page
        offset += limit

        # If we've processed all items (less than limit returned), exit the loop
        if len(items) < limit:
            break

    # Check results
    if not matching_resources:
        context = list_method_kwargs.get("project_id", "")
        context_msg = f" in project {context}" if context else ""
        raise not_found_error(
            f"No {resource_type_name} with name '{resource_name}' found{context_msg}"
        )

    if len(matching_resources) > 1:
        # Multiple matches found - construct an informative error message
        resource_ids = [r.get("id") for r in matching_resources]
        context = list_method_kwargs.get("project_id", "")
        context_msg = f" in project {context}" if context else ""

        raise duplicate_error(
            f"Multiple {resource_type_name}s found with name '{resource_name}'{context_msg}. "
            f"{resource_type_name.capitalize()} IDs: {', '.join(resource_ids)}. "
            f"Please use get_{resource_type_name}() with a specific ID instead."
        )

    # Exactly one match found - retrieve full details
    if "project_id" in list_method_kwargs:
        return await get_method(
            list_method_kwargs["project_id"], matching_resources[0].get("id")
        )
    else:
        return await get_method(matching_resources[0].get("id"))


# ---- Projects ----
@patch
async def list_projects(
    self: RagasApiClient,
    ids: t.Optional[t.List[str]] = None,
    limit: int = 50,
    offset: int = 0,
    order_by: t.Optional[str] = None,
    sort_dir: t.Optional[str] = None,
) -> t.Dict:
    """List projects."""
    params = {"limit": limit, "offset": offset}

    if ids:
        params["ids"] = ",".join(ids)

    if order_by:
        params["order_by"] = order_by

    if sort_dir:
        params["sort_dir"] = sort_dir

    return await self._list_resources("projects", **params)


@patch
async def get_project(self: RagasApiClient, project_id: str) -> t.Dict:
    """Get a specific project by ID."""
    # TODO: Need get project by title
    return await self._get_resource(f"projects/{project_id}")


@patch
async def create_project(
    self: RagasApiClient, title: str, description: t.Optional[str] = None
) -> t.Dict:
    """Create a new project."""
    data = {"title": title}
    if description:
        data["description"] = description
    return await self._create_resource("projects", data)


@patch
async def update_project(
    self: RagasApiClient,
    project_id: str,
    title: t.Optional[str] = None,
    description: t.Optional[str] = None,
) -> t.Dict:
    """Update an existing project."""
    data = {}
    if title:
        data["title"] = title
    if description:
        data["description"] = description
    return await self._update_resource(f"projects/{project_id}", data)


@patch
async def delete_project(self: RagasApiClient, project_id: str) -> None:
    """Delete a project."""
    await self._delete_resource(f"projects/{project_id}")


@patch
async def get_project_by_name(self: RagasApiClient, project_name: str) -> t.Dict:
    """Get a project by its name.

    Args:
        project_name: Name of the project to find

    Returns:
        The project information dictionary

    Raises:
        ProjectNotFoundError: If no project with the given name is found
        DuplicateProjectError: If multiple projects with the given name are found
    """
    return await self._get_resource_by_name(
        list_method=self.list_projects,
        get_method=self.get_project,
        resource_name=project_name,
        name_field="title",  # Projects use 'title' instead of 'name'
        not_found_error=ProjectNotFoundError,
        duplicate_error=DuplicateProjectError,
        resource_type_name="project",
    )


# ---- Datasets ----
@patch
async def list_datasets(
    self: RagasApiClient,
    project_id: str,
    limit: int = 50,
    offset: int = 0,
    order_by: t.Optional[str] = None,
    sort_dir: t.Optional[str] = None,
) -> t.Dict:
    """List datasets in a project."""
    params = {"limit": limit, "offset": offset}
    if order_by:
        params["order_by"] = order_by
    if sort_dir:
        params["sort_dir"] = sort_dir
    return await self._list_resources(f"projects/{project_id}/datasets", **params)


@patch
async def get_dataset(self: RagasApiClient, project_id: str, dataset_id: str) -> t.Dict:
    """Get a specific dataset."""
    return await self._get_resource(f"projects/{project_id}/datasets/{dataset_id}")


@patch
async def create_dataset(
    self: RagasApiClient,
    project_id: str,
    name: str,
    description: t.Optional[str] = None,
) -> t.Dict:
    """Create a new dataset in a project."""
    data = {"name": name}
    if description:
        data["description"] = description
    return await self._create_resource(f"projects/{project_id}/datasets", data)


@patch
async def update_dataset(
    self: RagasApiClient,
    project_id: str,
    dataset_id: str,
    name: t.Optional[str] = None,
    description: t.Optional[str] = None,
) -> t.Dict:
    """Update an existing dataset."""
    data = {}
    if name:
        data["name"] = name
    if description:
        data["description"] = description
    return await self._update_resource(
        f"projects/{project_id}/datasets/{dataset_id}", data
    )


@patch
async def delete_dataset(
    self: RagasApiClient, project_id: str, dataset_id: str
) -> None:
    """Delete a dataset."""
    await self._delete_resource(f"projects/{project_id}/datasets/{dataset_id}")


@patch
async def get_dataset_by_name(
    self: RagasApiClient, project_id: str, dataset_name: str
) -> t.Dict:
    """Get a dataset by its name.

    Args:
        project_id: ID of the project
        dataset_name: Name of the dataset to find

    Returns:
        The dataset information dictionary

    Raises:
        DatasetNotFoundError: If no dataset with the given name is found
        DuplicateDatasetError: If multiple datasets with the given name are found
    """
    return await self._get_resource_by_name(
        list_method=self.list_datasets,
        get_method=self.get_dataset,
        resource_name=dataset_name,
        name_field="name",
        not_found_error=DatasetNotFoundError,
        duplicate_error=DuplicateDatasetError,
        resource_type_name="dataset",
        project_id=project_id,
    )


# ---- Experiments ----
@patch
async def list_experiments(
    self: RagasApiClient,
    project_id: str,
    limit: int = 50,
    offset: int = 0,
    order_by: t.Optional[str] = None,
    sort_dir: t.Optional[str] = None,
) -> t.Dict:
    """List experiments in a project."""
    params = {"limit": limit, "offset": offset}
    if order_by:
        params["order_by"] = order_by
    if sort_dir:
        params["sort_dir"] = sort_dir
    return await self._list_resources(f"projects/{project_id}/experiments", **params)


@patch
async def get_experiment(
    self: RagasApiClient, project_id: str, experiment_id: str
) -> t.Dict:
    """Get a specific experiment."""
    return await self._get_resource(
        f"projects/{project_id}/experiments/{experiment_id}"
    )


@patch
async def create_experiment(
    self: RagasApiClient,
    project_id: str,
    name: str,
    description: t.Optional[str] = None,
) -> t.Dict:
    """Create a new experiment in a project."""
    data = {"name": name}
    if description:
        data["description"] = description
    return await self._create_resource(f"projects/{project_id}/experiments", data)


@patch
async def update_experiment(
    self: RagasApiClient,
    project_id: str,
    experiment_id: str,
    name: t.Optional[str] = None,
    description: t.Optional[str] = None,
) -> t.Dict:
    """Update an existing experiment."""
    data = {}
    if name:
        data["name"] = name
    if description:
        data["description"] = description
    return await self._update_resource(
        f"projects/{project_id}/experiments/{experiment_id}", data
    )


@patch
async def delete_experiment(
    self: RagasApiClient, project_id: str, experiment_id: str
) -> None:
    """Delete an experiment."""
    await self._delete_resource(f"projects/{project_id}/experiments/{experiment_id}")


@patch
async def get_experiment_by_name(
    self: RagasApiClient, project_id: str, experiment_name: str
) -> t.Dict:
    """Get an experiment by its name.

    Args:
        project_id: ID of the project containing the experiment
        experiment_name: Name of the experiment to find

    Returns:
        The experiment information dictionary

    Raises:
        ExperimentNotFoundError: If no experiment with the given name is found
        DuplicateExperimentError: If multiple experiments with the given name are found
    """
    return await self._get_resource_by_name(
        list_method=self.list_experiments,
        get_method=self.get_experiment,
        resource_name=experiment_name,
        name_field="name",
        not_found_error=ExperimentNotFoundError,
        duplicate_error=DuplicateExperimentError,
        resource_type_name="experiment",
        project_id=project_id,
    )


# ---- Dataset Columns ----
@patch
async def list_dataset_columns(
    self: RagasApiClient,
    project_id: str,
    dataset_id: str,
    limit: int = 50,
    offset: int = 0,
    order_by: t.Optional[str] = None,
    sort_dir: t.Optional[str] = None,
) -> t.Dict:
    """List columns in a dataset."""
    params = {"limit": limit, "offset": offset}
    if order_by:
        params["order_by"] = order_by
    if sort_dir:
        params["sort_dir"] = sort_dir
    return await self._list_resources(
        f"projects/{project_id}/datasets/{dataset_id}/columns", **params
    )


@patch
async def get_dataset_column(
    self: RagasApiClient, project_id: str, dataset_id: str, column_id: str
) -> t.Dict:
    """Get a specific column in a dataset."""
    return await self._get_resource(
        f"projects/{project_id}/datasets/{dataset_id}/columns/{column_id}"
    )


@patch
async def create_dataset_column(
    self: RagasApiClient,
    project_id: str,
    dataset_id: str,
    id: str,
    name: str,
    type: str,
    col_order: t.Optional[int] = None,
    settings: t.Optional[t.Dict] = None,
) -> t.Dict:
    """Create a new column in a dataset."""
    data = {"id": id, "name": name, "type": type}
    if col_order is not None:
        data["col_order"] = col_order
    if settings:
        data["settings"] = settings
    return await self._create_resource(
        f"projects/{project_id}/datasets/{dataset_id}/columns", data
    )


@patch
async def update_dataset_column(
    self: RagasApiClient,
    project_id: str,
    dataset_id: str,
    column_id: str,
    **column_data,
) -> t.Dict:
    """Update an existing column in a dataset."""
    return await self._update_resource(
        f"projects/{project_id}/datasets/{dataset_id}/columns/{column_id}",
        column_data,
    )


@patch
async def delete_dataset_column(
    self: RagasApiClient, project_id: str, dataset_id: str, column_id: str
) -> None:
    """Delete a column from a dataset."""
    await self._delete_resource(
        f"projects/{project_id}/datasets/{dataset_id}/columns/{column_id}"
    )


# ---- Dataset Rows ----
@patch
async def list_dataset_rows(
    self: RagasApiClient,
    project_id: str,
    dataset_id: str,
    limit: int = 50,
    offset: int = 0,
    order_by: t.Optional[str] = None,
    sort_dir: t.Optional[str] = None,
) -> t.Dict:
    """List rows in a dataset."""
    params = {"limit": limit, "offset": offset}
    if order_by:
        params["order_by"] = order_by
    if sort_dir:
        params["sort_dir"] = sort_dir
    return await self._list_resources(
        f"projects/{project_id}/datasets/{dataset_id}/rows", **params
    )


@patch
async def get_dataset_row(
    self: RagasApiClient, project_id: str, dataset_id: str, row_id: str
) -> t.Dict:
    """Get a specific row in a dataset."""
    return await self._get_resource(
        f"projects/{project_id}/datasets/{dataset_id}/rows/{row_id}"
    )


@patch
async def create_dataset_row(
    self: RagasApiClient, project_id: str, dataset_id: str, id: str, data: t.Dict
) -> t.Dict:
    """Create a new row in a dataset."""
    row_data = {"id": id, "data": data}
    return await self._create_resource(
        f"projects/{project_id}/datasets/{dataset_id}/rows", row_data
    )


@patch
async def update_dataset_row(
    self: RagasApiClient, project_id: str, dataset_id: str, row_id: str, data: t.Dict
) -> t.Dict:
    """Update an existing row in a dataset."""
    row_data = {"data": data}
    return await self._update_resource(
        f"projects/{project_id}/datasets/{dataset_id}/rows/{row_id}",
        row_data,
    )


@patch
async def delete_dataset_row(
    self: RagasApiClient, project_id: str, dataset_id: str, row_id: str
) -> None:
    """Delete a row from a dataset."""
    await self._delete_resource(
        f"projects/{project_id}/datasets/{dataset_id}/rows/{row_id}"
    )


def create_nano_id(size=12):
    # Define characters to use (alphanumeric)
    alphabet = string.ascii_letters + string.digits

    # Generate UUID and convert to int
    uuid_int = uuid.uuid4().int

    # Convert to base62
    result = ""
    while uuid_int:
        uuid_int, remainder = divmod(uuid_int, len(alphabet))
        result = alphabet[remainder] + result

    # Pad if necessary and return desired length
    return result[:size]


# Default settings for columns
DEFAULT_SETTINGS = {"is_required": False, "max_length": 1000}


# Model definitions
class Column(BaseModel):
    id: str = Field(default_factory=create_nano_id)
    name: str = Field(...)
    type: str = Field(...)
    settings: t.Dict = Field(default_factory=lambda: DEFAULT_SETTINGS.copy())
    col_order: t.Optional[int] = Field(default=None)


class RowCell(BaseModel):
    data: t.Any = Field(...)
    column_id: str = Field(...)


class Row(BaseModel):
    id: str = Field(default_factory=create_nano_id)
    data: t.List[RowCell] = Field(...)


# ---- Resource With Data Helper Methods ----
@patch
async def _create_with_data(
    self: RagasApiClient,
    resource_type: str,
    project_id: str,
    name: str,
    description: str,
    columns: t.List[Column],
    rows: t.List[Row],
    batch_size: int = 50,
) -> t.Dict:
    """Generic method to create a resource with columns and rows.

    Args:
        resource_type: Type of resource ("dataset" or "experiment")
        project_id: Project ID
        name: Resource name
        description: Resource description
        columns: List of column definitions
        rows: List of row data
        batch_size: Number of operations to perform concurrently

    Returns:
        The created resource
    """
    # Select appropriate methods based on resource type
    if resource_type == "dataset":
        create_fn = self.create_dataset
        create_col_fn = self.create_dataset_column
        create_row_fn = self.create_dataset_row
        delete_fn = self.delete_dataset
        id_key = "dataset_id"
    elif resource_type == "experiment":
        create_fn = self.create_experiment
        create_col_fn = self.create_experiment_column
        create_row_fn = self.create_experiment_row
        delete_fn = self.delete_experiment
        id_key = "experiment_id"
    else:
        raise ValueError(f"Unsupported resource type: {resource_type}")

    try:
        # Create the resource
        resource = await create_fn(project_id, name, description)

        # Process columns in batches
        for i in range(0, len(columns), batch_size):
            batch = columns[i : i + batch_size]
            col_tasks = []

            for col in batch:
                params = {
                    "project_id": project_id,
                    id_key: resource["id"],  # dataset_id here
                    "id": col.id,
                    "name": col.name,
                    "type": col.type,
                    "settings": col.settings,
                }
                if col.col_order is not None:
                    params["col_order"] = col.col_order

                col_tasks.append(create_col_fn(**params))

            await asyncio.gather(*col_tasks)

        # Process rows in batches
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            row_tasks = []

            for row in batch:
                row_data = {cell.column_id: cell.data for cell in row.data}
                row_tasks.append(
                    create_row_fn(
                        project_id=project_id,
                        **{id_key: resource["id"]},
                        id=row.id,
                        data=row_data,
                    )
                )

            await asyncio.gather(*row_tasks)

        return resource

    except Exception as e:
        # Clean up on error
        if "resource" in locals():
            try:
                await delete_fn(project_id, resource["id"])
            except Exception:
                pass  # Ignore cleanup errors
        raise e


@patch
async def create_dataset_with_data(
    self: RagasApiClient,
    project_id: str,
    name: str,
    description: str,
    columns: t.List[Column],
    rows: t.List[Row],
    batch_size: int = 50,
) -> t.Dict:
    """Create a dataset with columns and rows.

    This method creates a dataset and populates it with columns and rows in an
    optimized way using concurrent requests.

    Args:
        project_id: Project ID
        name: Dataset name
        description: Dataset description
        columns: List of column definitions
        rows: List of row data
        batch_size: Number of operations to perform concurrently

    Returns:
        The created dataset
    """
    return await self._create_with_data(
        "dataset", project_id, name, description, columns, rows, batch_size
    )


# ---- Experiment Columns ----
@patch
async def list_experiment_columns(
    self: RagasApiClient,
    project_id: str,
    experiment_id: str,
    limit: int = 50,
    offset: int = 0,
    order_by: t.Optional[str] = None,
    sort_dir: t.Optional[str] = None,
) -> t.Dict:
    """List columns in an experiment."""
    params = {"limit": limit, "offset": offset}
    if order_by:
        params["order_by"] = order_by
    if sort_dir:
        params["sort_dir"] = sort_dir
    return await self._list_resources(
        f"projects/{project_id}/experiments/{experiment_id}/columns", **params
    )


@patch
async def get_experiment_column(
    self: RagasApiClient, project_id: str, experiment_id: str, column_id: str
) -> t.Dict:
    """Get a specific column in an experiment."""
    return await self._get_resource(
        f"projects/{project_id}/experiments/{experiment_id}/columns/{column_id}"
    )


@patch
async def create_experiment_column(
    self: RagasApiClient,
    project_id: str,
    experiment_id: str,
    id: str,
    name: str,
    type: str,
    col_order: t.Optional[int] = None,
    settings: t.Optional[t.Dict] = None,
) -> t.Dict:
    """Create a new column in an experiment."""
    data = {"id": id, "name": name, "type": type}
    if col_order is not None:
        data["col_order"] = col_order
    if settings:
        data["settings"] = settings
    return await self._create_resource(
        f"projects/{project_id}/experiments/{experiment_id}/columns", data
    )


@patch
async def update_experiment_column(
    self: RagasApiClient,
    project_id: str,
    experiment_id: str,
    column_id: str,
    **column_data,
) -> t.Dict:
    """Update an existing column in an experiment."""
    return await self._update_resource(
        f"projects/{project_id}/experiments/{experiment_id}/columns/{column_id}",
        column_data,
    )


@patch
async def delete_experiment_column(
    self: RagasApiClient, project_id: str, experiment_id: str, column_id: str
) -> None:
    """Delete a column from an experiment."""
    await self._delete_resource(
        f"projects/{project_id}/experiments/{experiment_id}/columns/{column_id}"
    )


# ---- Experiment Rows ----
@patch
async def list_experiment_rows(
    self: RagasApiClient,
    project_id: str,
    experiment_id: str,
    limit: int = 50,
    offset: int = 0,
    order_by: t.Optional[str] = None,
    sort_dir: t.Optional[str] = None,
) -> t.Dict:
    """List rows in an experiment."""
    params = {"limit": limit, "offset": offset}
    if order_by:
        params["order_by"] = order_by
    if sort_dir:
        params["sort_dir"] = sort_dir
    return await self._list_resources(
        f"projects/{project_id}/experiments/{experiment_id}/rows", **params
    )


@patch
async def get_experiment_row(
    self: RagasApiClient, project_id: str, experiment_id: str, row_id: str
) -> t.Dict:
    """Get a specific row in an experiment."""
    return await self._get_resource(
        f"projects/{project_id}/experiments/{experiment_id}/rows/{row_id}"
    )


@patch
async def create_experiment_row(
    self: RagasApiClient, project_id: str, experiment_id: str, id: str, data: t.Dict
) -> t.Dict:
    """Create a new row in an experiment."""
    row_data = {"id": id, "data": data}
    return await self._create_resource(
        f"projects/{project_id}/experiments/{experiment_id}/rows", row_data
    )


@patch
async def update_experiment_row(
    self: RagasApiClient, project_id: str, experiment_id: str, row_id: str, data: t.Dict
) -> t.Dict:
    """Update an existing row in an experiment."""
    row_data = {"data": data}
    return await self._update_resource(
        f"projects/{project_id}/experiments/{experiment_id}/rows/{row_id}",
        row_data,
    )


@patch
async def delete_experiment_row(
    self: RagasApiClient, project_id: str, experiment_id: str, row_id: str
) -> None:
    """Delete a row from an experiment."""
    await self._delete_resource(
        f"projects/{project_id}/experiments/{experiment_id}/rows/{row_id}"
    )


@patch
async def create_experiment_with_data(
    self: RagasApiClient,
    project_id: str,
    name: str,
    description: str,
    columns: t.List[Column],
    rows: t.List[Row],
    batch_size: int = 50,
) -> t.Dict:
    """Create an experiment with columns and rows.

    This method creates an experiment and populates it with columns and rows in an
    optimized way using concurrent requests.

    Args:
        project_id: Project ID
        name: Experiment name
        description: Experiment description
        columns: List of column definitions
        rows: List of row data
        batch_size: Number of operations to perform concurrently

    Returns:
        The created experiment
    """
    return await self._create_with_data(
        "experiment", project_id, name, description, columns, rows, batch_size
    )


# ---- Utility Methods ----
@patch
def create_column(
    self: RagasApiClient,
    name: str,
    type: str,
    settings: t.Optional[t.Dict] = None,
    col_order: t.Optional[int] = None,
    id: t.Optional[str] = None,
) -> Column:
    """Create a Column object.

    Args:
        name: Column name
        type: Column type (use ColumnType enum)
        settings: Column settings
        col_order: Column order
        id: Custom ID (generates one if not provided)

    Returns:
        Column object
    """
    params = {"name": name, "type": type}
    if settings:
        params["settings"] = settings
    if col_order is not None:
        params["col_order"] = col_order
    if id:
        params["id"] = id

    return Column(**params)


@patch
def create_row(
    self: RagasApiClient,
    data: t.Dict[str, t.Any],
    column_map: t.Dict[str, str],
    id: t.Optional[str] = None,
) -> Row:
    """Create a Row object from a dictionary.

    Args:
        data: Dictionary mapping column names to values
        column_map: Dictionary mapping column names to column IDs
        id: Custom ID (generates one if not provided)

    Returns:
        Row object
    """
    cells = []
    for col_name, value in data.items():
        if col_name in column_map:
            cells.append(RowCell(data=value, column_id=column_map[col_name]))

    params = {"data": cells}
    if id:
        params["id"] = id

    return Row(**params)


@patch
def create_column_map(
    self: RagasApiClient, columns: t.List[Column]
) -> t.Dict[str, str]:
    """Create a mapping of column names to IDs.

    Args:
        columns: List of column objects

    Returns:
        Dictionary mapping column names to IDs
    """
    return {col.name: col.id for col in columns}


@patch
async def convert_raw_data(
    self: RagasApiClient, column_defs: t.List[t.Dict], row_data: t.List[t.Dict]
) -> t.Tuple[t.List[Column], t.List[Row]]:
    """Convert raw data to column and row objects.

    Args:
        column_defs: List of column definitions (dicts with name, type)
        row_data: List of dictionaries with row data

    Returns:
        Tuple of (columns, rows)
    """
    # Create columns
    columns = []
    for col in column_defs:
        columns.append(self.create_column(**col))

    # Create column map
    column_map = self.create_column_map(columns)

    # Create rows
    rows = []
    for data in row_data:
        rows.append(self.create_row(data, column_map))

    return columns, rows
