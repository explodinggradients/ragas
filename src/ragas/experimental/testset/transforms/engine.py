import typing as t

from ragas.experimental.testset.graph import KnowledgeGraph
from ragas.experimental.testset.transforms.base import (
    BaseGraphTransformations,
    Parallel,
    Sequences,
)


class TransformerEngine:
    def apply(
        self, transforms: t.List[BaseGraphTransformations], on: KnowledgeGraph
    ) -> KnowledgeGraph:
        pass

    def rollback(
        self, transforms: t.List[BaseGraphTransformations], on: KnowledgeGraph
    ) -> KnowledgeGraph:
        raise NotImplementedError
