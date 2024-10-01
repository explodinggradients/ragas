from __future__ import annotations

import asyncio
import logging
import typing as t

from ragas.executor import as_completed, is_event_loop_running, tqdm
from ragas.run_config import RunConfig
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.transforms.base import BaseGraphTransformation

logger = logging.getLogger(__name__)

Transforms = t.Union[
    t.List[BaseGraphTransformation],
    "Parallel",
    BaseGraphTransformation,
]


class Parallel:
    def __init__(self, *transformations: BaseGraphTransformation):
        self.transformations = list(transformations)

    def generate_execution_plan(self, kg: KnowledgeGraph) -> t.List[t.Coroutine]:
        coroutines = []
        for transformation in self.transformations:
            coroutines.extend(transformation.generate_execution_plan(kg))
        return coroutines


async def run_coroutines(coroutines: t.List[t.Coroutine], desc: str, max_workers: int):
    """
    Run a list of coroutines in parallel.
    """
    for future in tqdm(
        await as_completed(coroutines, max_workers=max_workers),
        desc=desc,
        total=len(coroutines),
        # whether you want to keep the progress bar after completion
        leave=False,
    ):
        try:
            await future
        except Exception as e:
            logger.error(f"unable to apply transformation: {e}")


def get_desc(transform: BaseGraphTransformation | Parallel):
    if isinstance(transform, Parallel):
        transform_names = [t.__class__.__name__ for t in transform.transformations]
        return f"Applying [{', '.join(transform_names)}]"
    else:
        return f"Applying {transform.__class__.__name__}"


def apply_nest_asyncio():
    NEST_ASYNCIO_APPLIED: bool = False
    if is_event_loop_running():
        # an event loop is running so call nested_asyncio to fix this
        try:
            import nest_asyncio
        except ImportError:
            raise ImportError(
                "It seems like your running this in a jupyter-like environment. Please install nest_asyncio with `pip install nest_asyncio` to make it work."
            )

        if not NEST_ASYNCIO_APPLIED:
            nest_asyncio.apply()
            NEST_ASYNCIO_APPLIED = True


def apply_transforms(
    kg: KnowledgeGraph,
    transforms: Transforms,
    run_config: RunConfig = RunConfig(),
):
    """
    Apply a list of transformations to a knowledge graph in place.
    """
    # apply nest_asyncio to fix the event loop issue in jupyter
    apply_nest_asyncio()

    # if single transformation, wrap it in a list
    if isinstance(transforms, BaseGraphTransformation):
        transforms = [transforms]

    # apply the transformations
    # if Sequences, apply each transformation sequentially
    if isinstance(transforms, t.List):
        for transform in transforms:
            asyncio.run(
                run_coroutines(
                    transform.generate_execution_plan(kg),
                    get_desc(transform),
                    run_config.max_workers,
                )
            )
    # if Parallel, collect inside it and run it all
    elif isinstance(transforms, Parallel):
        asyncio.run(
            run_coroutines(
                transforms.generate_execution_plan(kg),
                get_desc(transforms),
                run_config.max_workers,
            )
        )
    else:
        raise ValueError(
            f"Invalid transforms type: {type(transforms)}. Expects a list of BaseGraphTransformations or a Parallel instance."
        )


def rollback_transforms(kg: KnowledgeGraph, transforms: Transforms):
    # this will allow you to roll back the transformations
    raise NotImplementedError
