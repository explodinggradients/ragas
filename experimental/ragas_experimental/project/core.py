"""Use this class to represent the AI project that we are working on and to interact with datasets and experiments in it."""

__all__ = ["Project"]

import typing as t
import os

from fastcore.utils import patch

from ..backends.factory import RagasApiClientFactory
from ..backends.ragas_api_client import RagasApiClient
import ragas_experimental.typing as rt
from ..utils import async_to_sync


class Project:
    def __init__(
        self,
        project_id: str,
        backend: rt.SUPPORTED_BACKENDS = "local",
        root_dir: t.Optional[str] = None,
        ragas_api_client: t.Optional[RagasApiClient] = None,
    ):
        self.project_id = project_id
        self.backend = backend

        if backend == "local":
            if root_dir is None:
                raise ValueError("root_dir is required for local backend")
            self._root_dir = os.path.join(root_dir, project_id)
            # Ensure project directory structure exists
            self._create_local_project_structure()
        elif backend == "ragas_app":
            if ragas_api_client is None:
                self._ragas_api_client = RagasApiClientFactory.create()
            else:
                self._ragas_api_client = ragas_api_client
        else:
            raise ValueError(f"Invalid backend: {backend}")

        # Initialize project properties
        if backend == "ragas_app":
            try:
                sync_version = async_to_sync(self._ragas_api_client.get_project)
                existing_project = sync_version(project_id=self.project_id)
                self.project_id = existing_project["id"]
                self.name = existing_project["title"]
                self.description = existing_project["description"]
            except Exception as e:
                raise e
        elif backend == "local":
            self.name = self.project_id
            self.description = ""

    def _create_local_project_structure(self):
        """Create the local directory structure for the project"""
        os.makedirs(self._root_dir, exist_ok=True)
        # Create datasets directory
        os.makedirs(os.path.join(self._root_dir, "datasets"), exist_ok=True)
        # Create experiments directory
        os.makedirs(os.path.join(self._root_dir, "experiments"), exist_ok=True)


@patch(cls_method=True)
def create(
    cls: Project,
    name: str,
    description: str = "",
    backend: rt.SUPPORTED_BACKENDS = "local",
    root_dir: t.Optional[str] = None,
    ragas_api_client: t.Optional[RagasApiClient] = None,
):
    if backend == "ragas_app":
        ragas_api_client = ragas_api_client or RagasApiClientFactory.create()
        sync_version = async_to_sync(ragas_api_client.create_project)
        new_project = sync_version(title=name, description=description)
        return cls(
            new_project["id"], backend="ragas_api", ragas_api_client=ragas_api_client
        )
    elif backend == "local":
        if root_dir is None:
            raise ValueError("root_dir is required for local backend")
        # For local backend, we use the name as the project_id
        project_id = name
        return cls(project_id, backend="local", root_dir=root_dir)


# %% ../../nbs/api/project/core.ipynb 9
@patch
def delete(self: Project):
    if self.backend == "ragas_app":
        sync_version = async_to_sync(self._ragas_api_client.delete_project)
        sync_version(project_id=self.project_id)
        print("Project deleted from Ragas API!")
    elif self.backend == "local":
        import shutil

        # Caution: this deletes the entire project directory
        if os.path.exists(self._root_dir):
            shutil.rmtree(self._root_dir)
            print(f"Local project at {self._root_dir} deleted!")
        else:
            print(f"Local project at {self._root_dir} does not exist")

    @patch
    def __repr__(self: Project):
        return f"Project(name='{self.name}', backend='{self.backend}')"


# %% ../../nbs/api/project/core.ipynb 11
@patch(cls_method=True)
def get(
    cls: Project,
    name: str,
    backend: rt.SUPPORTED_BACKENDS = "local",
    root_dir: t.Optional[str] = None,
    ragas_api_client: t.Optional[RagasApiClient] = None,
) -> Project:
    """Get an existing project by name.

    Args:
        name: The name of the project to get
        backend: The backend to use (ragas_api or local)
        root_dir: The root directory for local backends
        ragas_api_client: Optional custom Ragas API client

    Returns:
        Project: The project instance
    """
    if backend == "ragas_app":
        # Search for project with given name in Ragas API
        if ragas_api_client is None:
            ragas_api_client = RagasApiClientFactory.create()

        # get the project by name
        sync_version = async_to_sync(ragas_api_client.get_project_by_name)
        project_info = sync_version(project_name=name)

        # Return Project instance
        return Project(
            project_id=project_info["id"],
            backend="ragas_app",
            ragas_api_client=ragas_api_client,
        )
    elif backend == "local":
        if root_dir is None:
            raise ValueError("root_dir is required for local backend")

        # For local backend, check if project directory exists
        project_path = os.path.join(root_dir, name)
        if not os.path.exists(project_path):
            raise ValueError(f"Local project '{name}' does not exist at {project_path}")

        # Return Project instance
        return Project(
            project_id=name,
            backend="local",
            root_dir=root_dir,
        )
    else:
        raise ValueError(f"Invalid backend: {backend}")


@patch
def get_dataset_path(self: Project, dataset_name: str) -> str:
    """Get the path to a dataset file in the local backend"""
    if self.backend != "local":
        raise ValueError("This method is only available for local backend")
    return os.path.join(self._root_dir, "datasets", f"{dataset_name}.csv")


@patch
def get_experiment_path(self: Project, experiment_name: str) -> str:
    """Get the path to an experiment file in the local backend"""
    if self.backend != "local":
        raise ValueError("This method is only available for local backend")
    return os.path.join(self._root_dir, "experiments", f"{experiment_name}.csv")
