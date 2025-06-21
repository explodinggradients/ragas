"""How to run experiments"""
__all__ = ["ExperimentProtocol"]

from tqdm import tqdm
from functools import wraps
import asyncio
@patch
def create_experiment(
    self: Project, name: str, model: t.Type[NotionModel]
) -> Experiment:
    """Create a new experiment view.

    Args:
        name: Name of the experiment
        model: Model class defining the experiment structure

    Returns:
        ExperimentView: View for managing experiment results
    """
    if self.experiments_page_id == "":
        raise ValueError("Experiments page ID is not set")

    # Collect all properties from model fields
    properties = {}
    for field_name, field in model._fields.items():
        properties.update(field._to_notion_property())

    # Create the database
    database_id = self._notion_backend.create_new_database(
        parent_page_id=self.experiments_page_id, title=name, properties=properties
    )

    return Experiment(
        name=name,
        model=model,
        database_id=database_id,
        notion_backend=self._notion_backend,
    )
@patch
def get_experiment(self: Project, name: str, model: t.Type[NotionModel]) -> Experiment:
    """Get an existing experiment by name."""
    if self.experiments_page_id == "":
        raise ValueError("Experiments page ID is not set")

    # Search for database with given name
    database_id = self._notion_backend.get_database_id(
        parent_page_id=self.experiments_page_id, name=name, return_multiple=False
    )

    return Experiment(
        name=name,
        model=model,
        database_id=database_id,
        notion_backend=self._notion_backend,
    )
@t.runtime_checkable
class ExperimentProtocol(t.Protocol):
    async def __call__(self, *args, **kwargs): ...
    async def run_async(self, name: str, dataset: Dataset): ...
# this one we have to clean up
from langfuse.decorators import observe
@patch
def experiment(
    self: Project, experiment_model: t.Type[NotionModel], name_prefix: str = ""
):
    """Decorator for creating experiment functions.

    Args:
        name_prefix: Optional prefix for experiment names

    Returns:
        Decorator function that wraps experiment functions
    """

    def decorator(func: t.Callable) -> ExperimentProtocol:
        @wraps(func)
        async def wrapped_experiment(*args, **kwargs):
            # wrap the function with langfuse observation so that it can be traced
            # and spans inside the function can be retrieved with sync_trace()
            observed_func = observe(name=f"{name_prefix}-{func.__name__}")(func)

            return await observed_func(*args, **kwargs)

        # Add run method to the wrapped function
        async def run_async(name: str, dataset: Dataset):
            # Create tasks for all items
            tasks = []
            for item in dataset:
                tasks.append(wrapped_experiment(item))

            # Use as_completed with tqdm for progress tracking
            results = []
            for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                result = await future
                # Add each result to experiment view as it completes
                results.append(result)

            # upload results to experiment view
            experiment_view = self.create_experiment(name=name, model=experiment_model)
            for result in results:
                experiment_view.append(result)

            return experiment_view

        wrapped_experiment.__setattr__("run_async", run_async)
        return t.cast(ExperimentProtocol, wrapped_experiment)

    return decorator
