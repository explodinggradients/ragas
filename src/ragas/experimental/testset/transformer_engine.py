import typing as t
from dataclasses import dataclass

from ragas.experimental.testset.graph import KnowledgeGraph
from ragas.experimental.testset.graph_transforms import BaseGraphTransformations


@dataclass
class TransformerEngine:
    def apply(
        self, transforms: t.List[BaseGraphTransformations], on: KnowledgeGraph
    ) -> KnowledgeGraph:
        pass

    def rollback(
        self, transforms: t.List[BaseGraphTransformations], on: KnowledgeGraph
    ) -> KnowledgeGraph:
        raise NotImplementedError
