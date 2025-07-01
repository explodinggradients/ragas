"""Use this class to represent the AI project that we are working on and to interact with datasets and experiments in it."""

__all__ = ["Project"]

import os
import shutil
import typing as t
from typing import overload, Literal, Optional

from ragas_experimental.model.pydantic_model import (
    ExtendedPydanticBaseModel as BaseModel,
)

from ..dataset import Dataset
from ..experiment import Experiment
from ..backends import ProjectBackend, create_project_backend
from .decorators import add_experiment_decorators

# Type-only imports for Box client protocol
if t.TYPE_CHECKING:
    from ..backends.config import BoxClientProtocol
    from ..backends.ragas_api_client import RagasApiClient
    from ..backends.local_csv import LocalCSVProjectBackend
    from ..backends.ragas_app import RagasAppProjectBackend
else:
    # Runtime imports for isinstance checks
    try:
        from ..backends.local_csv import LocalCSVProjectBackend
    except ImportError:
        LocalCSVProjectBackend = None
    try:
        from ..backends.ragas_app import RagasAppProjectBackend
    except ImportError:
        RagasAppProjectBackend = None


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

    # Type-safe overloads for different backend types
    @overload
    @classmethod
    def create(
        cls,
        name: str,
        backend_type: Literal["local/csv"],
        *,
        description: str = "",
        root_dir: str = "./ragas_data",
    ) -> "Project": ...

    @overload
    @classmethod
    def create(
        cls,
        name: str,
        backend_type: Literal["ragas/app"],
        *,
        description: str = "",
        api_key: Optional[str] = None,
        api_url: str = "https://api.ragas.io",
        timeout: int = 30,
        max_retries: int = 3,
        ragas_api_client: Optional["RagasApiClient"] = None,
    ) -> "Project": ...

    @overload
    @classmethod
    def create(
        cls,
        name: str,
        backend_type: Literal["box/csv"],
        *,
        description: str = "",
        client: "BoxClientProtocol",
        root_folder_id: str = "0",
    ) -> "Project": ...

    @classmethod
    def create(
        cls, name: str, backend_type: str, *, description: str = "", **kwargs
    ) -> "Project":
        """Create a new project with the specified backend.

        Args:
            name: Name of the project
            backend_type: Backend type ("local/csv", "ragas/app", or "box/csv")
            description: Description of the project
            **kwargs: Backend-specific configuration parameters

        Returns:
            Project: A new project instance

        Examples:
            >>> # Create a local project with type-safe parameters
            >>> project = Project.create(
            ...     "my_project",
            ...     backend_type="local/csv",
            ...     root_dir="/path/to/projects"
            ... )

            >>> # Create a ragas/app project
            >>> project = Project.create(
            ...     "my_project",
            ...     backend_type="ragas/app",
            ...     api_key="your_api_key"
            ... )

            >>> # Create a Box project
            >>> project = Project.create(
            ...     "my_project",
            ...     backend_type="box/csv",
            ...     client=authenticated_box_client,
            ...     root_folder_id="123456"
            ... )
        """
        # Use the registry-based approach for backend creation
        backend = create_project_backend(backend_type, **kwargs)

        # Create and return the Project instance
        return cls(
            project_id=name,  # Use name as project_id for simplicity
            project_backend=backend,
            name=name,
            description=description,
        )

    # Type-safe overloads for get_project
    @overload
    @classmethod
    def get(
        cls,
        name: str,
        backend_type: Literal["local/csv"],
        *,
        root_dir: str = "./ragas_data",
    ) -> "Project": ...

    @overload
    @classmethod
    def get(
        cls,
        name: str,
        backend_type: Literal["ragas/app"],
        *,
        api_key: Optional[str] = None,
        api_url: str = "https://api.ragas.io",
        timeout: int = 30,
        max_retries: int = 3,
        ragas_api_client: Optional["RagasApiClient"] = None,
    ) -> "Project": ...

    @overload
    @classmethod
    def get(
        cls,
        name: str,
        backend_type: Literal["box/csv"],
        *,
        client: "BoxClientProtocol",
        root_folder_id: str = "0",
    ) -> "Project": ...

    @classmethod
    def get(cls, name: str, backend_type: str, **kwargs) -> "Project":
        """Get an existing project by name.

        Args:
            name: Name of the project to retrieve
            backend_type: Backend type ("local/csv", "ragas/app", or "box/csv")
            **kwargs: Backend-specific configuration parameters

        Returns:
            Project: The existing project instance

        Examples:
            >>> # Get a local project
            >>> project = Project.get("my_project", backend_type="local/csv", root_dir="/path/to/projects")

            >>> # Get a ragas/app project
            >>> project = Project.get("my_project", backend_type="ragas/app", api_key="your_api_key")

            >>> # Get a Box project
            >>> project = Project.get("my_project", backend_type="box/csv", client=authenticated_box_client)
        """
        # Use the registry-based approach for backend creation
        backend = create_project_backend(backend_type, **kwargs)

        # For local backend, check if project actually exists
        if backend_type == "local/csv":
            import os

            root_dir = kwargs.get("root_dir", "./ragas_data")
            project_dir = os.path.join(root_dir, name)
            if not os.path.exists(project_dir):
                raise ValueError(f"Local project '{name}' does not exist in {root_dir}")

        # Get the existing project using the backend
        return cls(
            project_id=name,
            project_backend=backend,
            name=name,
            description="",  # Description will be loaded from backend if available
        )

    def delete(self):
        """Delete the project and all its data."""
        # Check if backend has a delete method, otherwise handle basic deletion
        if hasattr(self._backend, "delete_project"):
            # Backend provides its own deletion logic
            self._backend.delete_project(self.project_id)
            print("Project deleted!")
        elif hasattr(self._backend, "root_dir"):
            # Local backend - delete project directory
            project_dir = os.path.join(self._backend.root_dir, self.project_id)
            if os.path.exists(project_dir):
                shutil.rmtree(project_dir)
                print(f"Local project at {project_dir} deleted!")
            else:
                print(f"Local project at {project_dir} does not exist")
        else:
            print("Project deletion not supported by this backend")

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

        # Use the new Dataset.create() method for cleaner interface
        return Dataset.create(name, model, self, "datasets")

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
        # Use the new Dataset.get_dataset() method for cleaner interface
        return Dataset.get_dataset(dataset_name, model, self, "datasets")

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
        # Create experiment using backend
        experiment_id = self._backend.create_experiment(name, model)
        backend = self._backend.get_experiment_backend(experiment_id, name, model)

        # Return Experiment object for better UX
        return Experiment(
            name=name,
            model=model,
            project_id=self.project_id,
            experiment_id=experiment_id,
            backend=backend,
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
        # Get experiment using backend
        experiment_id, experiment_backend = self._backend.get_experiment_by_name(
            experiment_name, model
        )

        # Return Experiment object for better UX
        return Experiment(
            name=experiment_name,
            model=model,
            project_id=self.project_id,
            experiment_id=experiment_id,
            backend=experiment_backend,
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
        if LocalCSVProjectBackend is None or not isinstance(
            self._backend, LocalCSVProjectBackend
        ):
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
        if LocalCSVProjectBackend is None or not isinstance(
            self._backend, LocalCSVProjectBackend
        ):
            raise ValueError("This method is only available for local/csv backend")
        return os.path.join(
            self._backend._project_dir, "experiments", f"{experiment_name}.csv"
        )

    def __repr__(self) -> str:
        """String representation of the project."""
        backend_name = (
            "ragas/app"
            if RagasAppProjectBackend is not None
            and isinstance(self._backend, RagasAppProjectBackend)
            else "local/csv"
        )
        return f"Project(name='{self.name}', backend='{backend_name}')"
