"""Experiments hold the results of an experiment against a dataset."""

__all__ = ["Experiment", "experiment", "version_experiment"]

import typing as t

import asyncio
import inspect
from pathlib import Path

import git
from tqdm import tqdm

from pydantic import BaseModel

from .backends.base import BaseBackend
from .dataset import Dataset, DataTable
from .utils import memorable_names, find_git_root


class Experiment(DataTable):
    DATATABLE_TYPE = "Experiment"


def version_experiment(
    experiment_name: str,
    commit_message: t.Optional[str] = None,
    repo_path: t.Union[str, Path, None] = None,
    create_branch: bool = True,
    stage_all: bool = False,
) -> str:
    """Version control the current state of the codebase for an experiment."""
    # Default to current directory if no repo path is provided
    if repo_path is None:
        repo_path = find_git_root()

    # Initialize git repo object
    repo = git.Repo(repo_path)

    # Check if there are any changes to the repo
    has_changes = False
    if stage_all and repo.is_dirty(untracked_files=True):
        print("Staging all changes")
        repo.git.add(".")
        has_changes = True
    elif repo.is_dirty(untracked_files=False):
        print("Staging changes to tracked files")
        repo.git.add("-u")
        has_changes = True

    # Check if there are uncommitted changes
    if has_changes:
        # Default commit message if none provided
        if commit_message is None:
            commit_message = f"Experiment: {experiment_name}"

        # Commit changes
        commit = repo.index.commit(commit_message)
        commit_hash = commit.hexsha
        print(f"Changes committed with hash: {commit_hash[:8]}")
    else:
        # No changes to commit, use current HEAD
        commit_hash = repo.head.commit.hexsha
        print("No changes detected, nothing to commit")

    # Format the branch/tag name
    version_name = f"ragas/{experiment_name}"

    # Create branch if requested
    if create_branch:
        repo.create_head(version_name, commit_hash)
        print(f"Created branch: {version_name}")

    return commit_hash


@t.runtime_checkable
class ExperimentProtocol(t.Protocol):
    async def __call__(self, *args, **kwargs) -> t.Any: ...
    async def arun(
        self,
        dataset: Dataset,
        name: t.Optional[str] = None,
        backend: t.Optional[t.Union[BaseBackend, str]] = None,
        *args,
        **kwargs,
    ) -> "Experiment": ...


class ExperimentWrapper:
    """Wrapper class that implements ExperimentProtocol for decorated functions."""

    def __init__(
        self,
        func: t.Callable,
        experiment_model: t.Optional[t.Type[BaseModel]] = None,
        default_backend: t.Optional[t.Union[BaseBackend, str]] = None,
        name_prefix: str = "",
    ):
        self.func = func
        self.experiment_model = experiment_model
        self.default_backend = default_backend
        self.name_prefix = name_prefix
        # Store function signature for validation
        self.signature = inspect.signature(func)
        # Preserve function metadata
        self.__name__ = getattr(func, "__name__", "experiment_function")
        self.__doc__ = getattr(func, "__doc__", None)

    def _validate_function_parameters(self, *args, **kwargs) -> None:
        """Validate that the function can be called with the provided arguments."""
        try:
            # Try to bind the arguments to the function signature
            self.signature.bind(*args, **kwargs)
        except TypeError as e:
            func_name = getattr(self.func, "__name__", "experiment_function")

            param_info = []
            for name, param in self.signature.parameters.items():
                if param.default == inspect.Parameter.empty:
                    param_info.append(f"{name} (required)")
                else:
                    param_info.append(f"{name} (optional)")

            expected_params = ", ".join(param_info)

            raise ValueError(
                f"Parameter validation failed for experiment function '{func_name}()'. "
                f"Expected parameters: [{expected_params}]. "
                f"Original error: {str(e)}"
            ) from e

    async def __call__(self, *args, **kwargs) -> t.Any:
        """Call the original function."""
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(*args, **kwargs)
        else:
            return self.func(*args, **kwargs)

    async def arun(
        self,
        dataset: Dataset,
        name: t.Optional[str] = None,
        backend: t.Optional[t.Union[BaseBackend, str]] = None,
        *args,
        **kwargs,
    ) -> "Experiment":
        """Run the experiment against a dataset."""
        # Validate function parameters before any setup
        # Use the first dataset item as a representative sample for validation
        if len(dataset) > 0:
            sample_item = next(iter(dataset))
            self._validate_function_parameters(sample_item, *args, **kwargs)

        # Generate name if not provided
        if name is None:
            name = memorable_names.generate_unique_name()
        if self.name_prefix:
            name = f"{self.name_prefix}-{name}"

        # Resolve backend
        experiment_backend = backend or self.default_backend
        if experiment_backend:
            resolved_backend = Experiment._resolve_backend(experiment_backend)
        else:
            resolved_backend = dataset.backend

        # Create experiment
        experiment_view = Experiment(
            name=name,
            data_model=self.experiment_model,
            backend=resolved_backend,
        )

        # Create tasks for all items
        tasks = []
        for item in dataset:
            tasks.append(self(item, *args, **kwargs))

        progress_bar = None
        try:
            progress_bar = tqdm(total=len(dataset), desc="Running experiment")

            # Process all items
            for future in asyncio.as_completed(tasks):
                try:
                    result = await future
                    if result is not None:
                        experiment_view.append(result)
                except Exception as e:
                    # Log individual task failures but continue
                    print(f"Warning: Task failed with error: {e}")
                finally:
                    progress_bar.update(1)

        finally:
            if progress_bar:
                progress_bar.close()

        # Save experiment
        experiment_view.save()

        return experiment_view


def experiment(
    experiment_model: t.Optional[t.Type[BaseModel]] = None,
    backend: t.Optional[t.Union[BaseBackend, str]] = None,
    name_prefix: str = "",
) -> t.Callable[[t.Callable], ExperimentProtocol]:
    """Decorator for creating experiment functions.

    Args:
        experiment_model: The Pydantic model type to use for experiment results
        backend: Optional backend to use for storing experiment results
        name_prefix: Optional prefix for experiment names

    Returns:
        Decorator function that wraps experiment functions

    Example:
        @experiment(ExperimentDataRow)
        async def run_experiment(row: TestDataRow):
            # experiment logic here
            return ExperimentDataRow(...)
    """

    def decorator(func: t.Callable) -> ExperimentProtocol:
        wrapper = ExperimentWrapper(
            func=func,
            experiment_model=experiment_model,
            default_backend=backend,
            name_prefix=name_prefix,
        )
        return t.cast(ExperimentProtocol, wrapper)

    return decorator
