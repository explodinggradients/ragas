"""Experiments hold the results of an experiment against a dataset."""

__all__ = ["Experiment"]

import typing as t

import asyncio
import os
from functools import wraps
from pathlib import Path

import git
from tqdm import tqdm

from ragas_experimental.dataset import Dataset, DataTable
from ragas_experimental.utils import async_to_sync, memorable_names, find_git_root


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
    async def __call__(self, *args, **kwargs): ...
    async def run_async(
        self, dataset: Dataset, name: t.Optional[str] = None, **kwargs
    ): ...


class ExperimentDecorator:
    """Base class for experiment decorators that adds methods to Project instances."""

    def __init__(self, project):
        self.project = project

    def experiment(
        self,
        experiment_model,
        name_prefix: str = "",
        save_to_git: bool = False,
        stage_all: bool = False,
    ):
        """Decorator for creating experiment functions.

        Args:
            experiment_model: The model type to use for experiment results
            name_prefix: Optional prefix for experiment names
            save_to_git: Whether to save experiment state to git
            stage_all: Whether to stage all files when saving to git

        Returns:
            Decorator function that wraps experiment functions
        """

        def decorator(func: t.Callable) -> ExperimentProtocol:
            @wraps(func)
            async def wrapped_experiment(*args, **kwargs):
                # Simply call the function
                return await func(*args, **kwargs)

            # Add run method to the wrapped function
            async def run_async(
                dataset: Dataset,
                name: t.Optional[str] = None,
                save_to_git: bool = save_to_git,
                stage_all: bool = stage_all,
            ):
                # If name is not provided, generate a memorable name
                if name is None:
                    name = memorable_names.generate_unique_name()
                if name_prefix:
                    name = f"{name_prefix}-{name}"

                experiment_view = None
                try:
                    # Create the experiment view
                    experiment_view = self.project.create_experiment(
                        name=name, model=experiment_model
                    )

                    # Create tasks for all items
                    tasks = []
                    for item in dataset:
                        tasks.append(wrapped_experiment(item))

                    # Calculate total operations (processing + appending)
                    total_operations = (
                        len(tasks) * 2
                    )  # Each item requires processing and appending

                    # Use tqdm for combined progress tracking
                    results = []
                    progress_bar = tqdm(
                        total=total_operations, desc="Running experiment"
                    )

                    # Process all items
                    for future in asyncio.as_completed(tasks):
                        result = await future
                        if result is not None:
                            results.append(result)
                        progress_bar.update(1)  # Update for task completion

                    # Append results to experiment view
                    for result in results:
                        experiment_view.append(result)
                        progress_bar.update(1)  # Update for append operation

                    progress_bar.close()

                except Exception as e:
                    # Clean up the experiment if there was an error and it was created
                    if experiment_view is not None:
                        try:
                            # For platform backend, delete via API
                            if hasattr(self.project._backend, "ragas_api_client"):
                                sync_version = async_to_sync(
                                    self.project._backend.ragas_api_client.delete_experiment
                                )
                                sync_version(
                                    project_id=self.project.project_id,
                                    experiment_id=experiment_view.experiment_id,
                                )
                            else:
                                # For local backend, delete the file
                                experiment_path = self.project.get_experiment_path(
                                    experiment_view.name
                                )
                                if os.path.exists(experiment_path):
                                    os.remove(experiment_path)
                        except Exception as cleanup_error:
                            print(
                                f"Failed to clean up experiment after error: {cleanup_error}"
                            )

                    # Re-raise the original exception
                    raise e

                # Save to git if requested
                if save_to_git:
                    repo_path = find_git_root()
                    version_experiment(
                        experiment_name=name, repo_path=repo_path, stage_all=stage_all
                    )

                return experiment_view

            wrapped_experiment.__setattr__("run_async", run_async)
            return t.cast(ExperimentProtocol, wrapped_experiment)

        return decorator
