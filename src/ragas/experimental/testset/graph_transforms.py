import typing as t
from abc import ABC, abstractmethod

from ragas.experimental.testset.graph import KnowledgeGraph, Node, Relationship


class BaseGraphTransformations:
    # extractor
    def create_properties(self, kg: KnowledgeGraph) -> t.List[t.Tuple[str, t.Any]]:
        raise NotImplementedError

    # splitter
    def create_nodes(
        self, kg: KnowledgeGraph
    ) -> t.Tuple[t.List[Node], t.List[Relationship]]:
        raise NotImplementedError

    # relationship builder
    def create_relationships(self, kg: KnowledgeGraph) -> t.List[Relationship]:
        raise NotImplementedError


class Extractor(BaseGraphTransformations):
    @abstractmethod
    def create_properties(self, kg: KnowledgeGraph) -> t.List[t.Tuple[str, t.Any]]:
        pass


class Splitter(BaseGraphTransformations):
    @abstractmethod
    def create_nodes(
        self, kg: KnowledgeGraph
    ) -> t.Tuple[t.List[Node], t.List[Relationship]]:
        pass


class RelationshipBuilder(BaseGraphTransformations):
    @abstractmethod
    def create_relationships(self, kg: KnowledgeGraph) -> t.List[Relationship]:
        pass
