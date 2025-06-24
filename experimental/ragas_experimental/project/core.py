"""Use this class to represent the AI project that we are working on and to interact with datasets and experiments in it."""

__all__ = ["Project"]

import os
import shutil
import typing as t

import ragas_experimental.typing as rt
from ragas_experimental.model.pydantic_model import (
    ExtendedPydanticBaseModel as BaseModel,
)

from ..backends.factory import RagasApiClientFactory
from ..backends.ragas_api_client import RagasApiClient
from ..dataset import Dataset
from ..experiment import Experiment
from ..utils import async_to_sync
from .backends import ProjectBackend
from .backends.local_csv import LocalCSVProjectBackend
from .backends.platform import PlatformProjectBackend
from .decorators import add_experiment_decorators


class Project:
    """Represents an AI project for managing datasets and experiments."""

    def __init__(
        self,
        project_id: str,
        project_backend: ProjectBackend,
        name: t.Optional[str] = None,
        description: t.Optional[str] = None,
    ):
        """Initialize a Project with a backend.

        Args:
            project_id: Unique identifier for the project
            project_backend: Backend instance for project operations
            name: Human-readable name for the project
            description: Optional description of the project
        """
        self.project_id = project_id
        self._backend = project_backend
        self.name = name or project_id
        self.description = description or ""

        # Initialize the backend with project information
        self._backend.initialize(project_id)

        # Add experiment decorator methods
        add_experiment_decorators(self)

    @classmethod
    def create(
        cls,
        name: str,
        description: str = "",
        backend: rt.SUPPORTED_BACKENDS = "local/csv",
        root_dir: t.Optional[str] = None,
        ragas_api_client: t.Optional[RagasApiClient] = None,
    ) -> "Project":
        """Create a new project.

        Args:
            name: Name of the project
            description: Description of the project
            backend: Backend type ("local/csv" or "ragas/app")
            root_dir: Root directory for local backends
            ragas_api_client: API client for ragas/app backend

        Returns:
            Project: A new project instance
        """
        if backend == "ragas/app":
            ragas_api_client = ragas_api_client or RagasApiClientFactory.create()
            sync_version = async_to_sync(ragas_api_client.create_project)
            new_project = sync_version(title=name, description=description)

            project_backend = PlatformProjectBackend(ragas_api_client)
            return cls(
                project_id=new_project["id"],
                project_backend=project_backend,
                name=new_project["title"],
                description=new_project["description"],
            )
        elif backend == "local/csv":
            if root_dir is None:
                raise ValueError("root_dir is required for local/csv backend")

            project_backend = LocalCSVProjectBackend(root_dir)
            return cls(
                project_id=name,  # Use name as project_id for local
                project_backend=project_backend,
                name=name,
                description=description,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    @classmethod
    def get(
        cls,
        name: str,
        backend: rt.SUPPORTED_BACKENDS = "local/csv",
        root_dir: t.Optional[str] = None,
        ragas_api_client: t.Optional[RagasApiClient] = None,
    ) -> "Project":
        """Get an existing project by name.

        Args:
            name: The name of the project to get
            backend: The backend to use ("local/csv" or "ragas/app")
            root_dir: The root directory for local backends
            ragas_api_client: Optional custom Ragas API client

        Returns:
            Project: The project instance
        """
        if backend == "ragas/app":
            if ragas_api_client is None:
                ragas_api_client = RagasApiClientFactory.create()

            # Get the project by name
            sync_version = async_to_sync(ragas_api_client.get_project_by_name)
            project_info = sync_version(project_name=name)

            project_backend = PlatformProjectBackend(ragas_api_client)
            return cls(
                project_id=project_info["id"],
                project_backend=project_backend,
                name=project_info["title"],
                description=project_info["description"],
            )
        elif backend == "local/csv":
            if root_dir is None:
                raise ValueError("root_dir is required for local/csv backend")

            # For local backend, check if project directory exists
            project_path = os.path.join(root_dir, name)
            if not os.path.exists(project_path):
                raise ValueError(
                    f"Local project '{name}' does not exist at {project_path}"
                )

            project_backend = LocalCSVProjectBackend(root_dir)
            return cls(
                project_id=name,
                project_backend=project_backend,
                name=name,
                description="",
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def delete(self):
        """Delete the project and all its data."""
        if isinstance(self._backend, PlatformProjectBackend):
            sync_version = async_to_sync(self._backend.ragas_api_client.delete_project)
            sync_version(project_id=self.project_id)
            print("Project deleted from Ragas platform!")
        elif isinstance(self._backend, LocalCSVProjectBackend):
            # Caution: this deletes the entire project directory
            project_dir = os.path.join(self._backend.root_dir, self.project_id)
            if os.path.exists(project_dir):
                shutil.rmtree(project_dir)
                print(f"Local project at {project_dir} deleted!")
            else:
                print(f"Local project at {project_dir} does not exist")

    # Dataset operations
    def create_dataset(
        self,
        model: t.Type[BaseModel],
        name: t.Optional[str] = None,
    ) -> Dataset:
        """Create a new dataset.

        Args:
            model: Model class defining the dataset structure
            name: Name of the dataset (defaults to model name if not provided)

        Returns:
            Dataset: A new dataset object for managing entries
        """
        if name is None:
            name = model.__name__

        dataset_id = self._backend.create_dataset(name, model)

        backend_name = (
            "ragas/app"
            if isinstance(self._backend, PlatformProjectBackend)
            else "local/csv"
        )

        return Dataset(
            name=name,
            model=model,
            project_id=self.project_id,
            dataset_id=dataset_id,
            datatable_type="datasets",
            ragas_api_client=getattr(self._backend, "ragas_api_client", None),
            backend=backend_name,
            local_root_dir=getattr(self._backend, "root_dir", None),
        )

    def get_dataset(
        self,
        dataset_name: str,
        model: t.Type[BaseModel],
    ) -> Dataset:
        """Get an existing dataset by name.

        Args:
            dataset_name: The name of the dataset to retrieve
            model: The model class to use for the dataset entries

        Returns:
            Dataset: The retrieved dataset
        """
        dataset_id, dataset_backend = self._backend.get_dataset_by_name(
            dataset_name, model
        )

        backend_name = (
            "ragas/app"
            if isinstance(self._backend, PlatformProjectBackend)
            else "local/csv"
        )

        return Dataset(
            name=dataset_name,
            model=model,
            project_id=self.project_id,
            dataset_id=dataset_id,
            datatable_type="datasets",
            ragas_api_client=getattr(self._backend, "ragas_api_client", None),
            backend=backend_name,
            local_root_dir=getattr(self._backend, "root_dir", None),
        )

    def list_datasets(self) -> t.List[str]:
        """List all datasets in the project.

        Returns:
            List[str]: Names of all datasets in the project
        """
        datasets = self._backend.list_datasets()
        return [dataset["name"] for dataset in datasets]

    # Experiment operations
    def create_experiment(
        self,
        name: str,
        model: t.Type[BaseModel],
    ) -> Experiment:
        """Create a new experiment.

        Args:
            name: Name of the experiment
            model: Model class defining the experiment structure

        Returns:
            Experiment: An experiment object for managing results
        """
        experiment_id = self._backend.create_experiment(name, model)

        backend_name = (
            "ragas/app"
            if isinstance(self._backend, PlatformProjectBackend)
            else "local/csv"
        )

        return Experiment(
            name=name,
            model=model,
            project_id=self.project_id,
            experiment_id=experiment_id,
            ragas_api_client=getattr(self._backend, "ragas_api_client", None),
            backend=backend_name,
            local_root_dir=getattr(self._backend, "root_dir", None),
        )

    def get_experiment(
        self,
        experiment_name: str,
        model: t.Type[BaseModel],
    ) -> Experiment:
        """Get an existing experiment by name.

        Args:
            experiment_name: The name of the experiment to retrieve
            model: The model class to use for the experiment results

        Returns:
            Experiment: The retrieved experiment
        """
        experiment_id, experiment_backend = self._backend.get_experiment_by_name(
            experiment_name, model
        )

        backend_name = (
            "ragas/app"
            if isinstance(self._backend, PlatformProjectBackend)
            else "local/csv"
        )

        return Experiment(
            name=experiment_name,
            model=model,
            project_id=self.project_id,
            experiment_id=experiment_id,
            ragas_api_client=getattr(self._backend, "ragas_api_client", None),
            backend=backend_name,
            local_root_dir=getattr(self._backend, "root_dir", None),
        )

    def list_experiments(self) -> t.List[str]:
        """List all experiments in the project.

        Returns:
            List[str]: Names of all experiments in the project
        """
        experiments = self._backend.list_experiments()
        return [experiment["name"] for experiment in experiments]

    # Utility methods for local backend compatibility
    def get_dataset_path(self, dataset_name: str) -> str:
        """Get the path to a dataset file in the local backend.

        Args:
            dataset_name: Name of the dataset

        Returns:
            str: Path to the dataset CSV file

        Raises:
            ValueError: If not using local backend
        """
        if not isinstance(self._backend, LocalCSVProjectBackend):
            raise ValueError("This method is only available for local/csv backend")
        return os.path.join(
            self._backend._project_dir, "datasets", f"{dataset_name}.csv"
        )

    def get_experiment_path(self, experiment_name: str) -> str:
        """Get the path to an experiment file in the local backend.

        Args:
            experiment_name: Name of the experiment

        Returns:
            str: Path to the experiment CSV file

        Raises:
            ValueError: If not using local backend
        """
        if not isinstance(self._backend, LocalCSVProjectBackend):
            raise ValueError("This method is only available for local/csv backend")
        return os.path.join(
            self._backend._project_dir, "experiments", f"{experiment_name}.csv"
        )

    def __repr__(self) -> str:
        """String representation of the project."""
        backend_name = (
            "ragas/app"
            if isinstance(self._backend, PlatformProjectBackend)
            else "local/csv"
        )
        return f"Project(name='{self.name}', backend='{backend_name}')"
