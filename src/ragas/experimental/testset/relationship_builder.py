import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ragas.experimental.testset.graph import KnowledgeGraph, Relationship


@dataclass
class RelationshipBuilder(ABC):
    @abstractmethod
    def build(self, graph: KnowledgeGraph) -> t.List[Relationship]:
        pass


@dataclass
class JaccardSimilarityBuilder(RelationshipBuilder):
    type: t.Optional[str] = None
    threshold: t.Optional[int] = 80
    attribute: t.Optional[str] = None

    def __post_init__(self):
        if self.type == "fuzzy":
            try:
                from fuzzywuzzy import fuzz
            except ImportError:
                raise ImportError(
                    "fuzzywuzzy is not installed. Run pip install fuzzywuzzy"
                )
            self.fuzz = fuzz

    def build(self, graph: KnowledgeGraph) -> t.List[Relationship]:
        pass


@dataclass
class CosineSimilarityBuilder(RelationshipBuilder):
    attribute: t.Optional[str] = None

    def build(self, graph: KnowledgeGraph) -> t.List[Relationship]:
        pass
