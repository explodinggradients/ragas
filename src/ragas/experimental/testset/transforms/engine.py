from __future__ import annotations

import asyncio
import logging
import typing as t
from dataclasses import dataclass

from ragas.executor import as_completed, is_event_loop_running, tqdm
from ragas.experimental.testset.graph import KnowledgeGraph
from ragas.experimental.testset.transforms.base import (
    BaseGraphTransformations,
)
from ragas.run_config import RunConfig

logger = logging.getLogger(__name__)


class Parallel:
    def __init__(self, *transformations: BaseGraphTransformations):
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
        leave=True,
    ):
        try:
            await future
        except Exception as e:
            logger.error(f"unable to apply transformation: {e}")


def get_desc(transform: BaseGraphTransformations | Parallel):
    if isinstance(transform, Parallel):
        transform_names = [t.__class__.__name__ for t in transform.transformations]
        return f"Applying [{', '.join(transform_names)}] transformations in parallel"
    else:
        return f"Applying {transform.__class__.__name__}"


@dataclass
class TransformerEngine:
    _nest_asyncio_applied: bool = False

    def _apply_nest_asyncio(self):
        if is_event_loop_running():
            # an event loop is running so call nested_asyncio to fix this
            try:
                import nest_asyncio
            except ImportError:
                raise ImportError(
                    "It seems like your running this in a jupyter-like environment. Please install nest_asyncio with `pip install nest_asyncio` to make it work."
                )

            if not self._nest_asyncio_applied:
                nest_asyncio.apply()
                self._nest_asyncio_applied = True

    def apply(
        self,
        transforms: t.List[BaseGraphTransformations] | Parallel,
        kg: KnowledgeGraph,
        run_config: RunConfig = RunConfig(),
    ) -> KnowledgeGraph:
        # apply nest_asyncio to fix the event loop issue in jupyter
        self._apply_nest_asyncio()

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

        return kg

    def rollback(
        self, transforms: t.List[BaseGraphTransformations], on: KnowledgeGraph
    ) -> KnowledgeGraph:
        # this will allow you to roll back the transformations
        raise NotImplementedError
