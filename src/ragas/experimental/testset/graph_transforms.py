import typing as t
from abc import ABC, abstractmethod

from attr import dataclass

from ragas.experimental.testset.graph import KnowledgeGraph, Node, Relationship


class BaseGraphTransformations(ABC):
    """
    Abstract base class for graph transformations on a KnowledgeGraph.
    """

    @abstractmethod
    async def transform(self, kg: KnowledgeGraph) -> t.Any:
        """
        Abstract method to transform the KnowledgeGraph. Transformations should be
        idempotent, meaning that applying the transformation multiple times should
        yield the same result as applying it once.

        Parameters
        ----------
        kg : KnowledgeGraph
            The knowledge graph to be transformed.

        Returns
        -------
        t.Any
            The transformed knowledge graph.
        """
        pass

    def filter(self, kg: KnowledgeGraph) -> KnowledgeGraph:
        """
        Filters the KnowledgeGraph and returns the filtered graph.

        Parameters
        ----------
        kg : KnowledgeGraph
            The knowledge graph to be filtered.

        Returns
        -------
        KnowledgeGraph
            The filtered knowledge graph.
        """
        return kg


class Extractor(BaseGraphTransformations):
    """
    Abstract base class for extractors that transform a KnowledgeGraph by extracting
    specific properties from its nodes.

    Methods
    -------
    transform(kg: KnowledgeGraph) -> t.List[t.Tuple[Node, t.Tuple[str, t.Any]]]
        Transforms the KnowledgeGraph by extracting properties from its nodes.

    extract(node: Node) -> t.Tuple[str, t.Any]
        Abstract method to extract a specific property from a node.
    """

    async def transform(
        self, kg: KnowledgeGraph
    ) -> t.List[t.Tuple[Node, t.Tuple[str, t.Any]]]:
        """
        Transforms the KnowledgeGraph by extracting properties from its nodes. Uses
        the `filter` method to filter the graph and the `extract` method to extract
        properties from each node.

        Parameters
        ----------
        kg : KnowledgeGraph
            The knowledge graph to be transformed.

        Returns
        -------
        t.List[t.Tuple[Node, t.Tuple[str, t.Any]]]
            A list of tuples where each tuple contains a node and the extracted
            property.

        Examples
        --------
        >>> kg = KnowledgeGraph(nodes=[Node(id=1, properties={"name": "Node1"}), Node(id=2, properties={"name": "Node2"})])
        >>> extractor = SomeConcreteExtractor()
        >>> extractor.transform(kg)
        [(Node(id=1, properties={"name": "Node1"}), ("property_name", "extracted_value")),
         (Node(id=2, properties={"name": "Node2"}), ("property_name", "extracted_value"))]
        """
        filtered = self.filter(kg)
        return [(node, await self.extract(node)) for node in filtered.nodes]

    @abstractmethod
    async def extract(self, node: Node) -> t.Tuple[str, t.Any]:
        """
        Abstract method to extract a specific property from a node.

        Parameters
        ----------
        node : Node
            The node from which to extract the property.

        Returns
        -------
        t.Tuple[str, t.Any]
            A tuple containing the property name and the extracted value.
        """
        pass


class Splitter(BaseGraphTransformations):
    """
    Abstract base class for splitters that transform a KnowledgeGraph by splitting
    its nodes into smaller chunks.

    Methods
    -------
    transform(kg: KnowledgeGraph) -> t.Tuple[t.List[Node], t.List[Relationship]]
        Transforms the KnowledgeGraph by splitting its nodes into smaller chunks.

    split(node: Node) -> t.Tuple[t.List[Node], t.List[Relationship]]
        Abstract method to split a node into smaller chunks.
    """

    async def transform(
        self, kg: KnowledgeGraph
    ) -> t.Tuple[t.List[Node], t.List[Relationship]]:
        """
        Transforms the KnowledgeGraph by splitting its nodes into smaller chunks.

        Parameters
        ----------
        kg : KnowledgeGraph
            The knowledge graph to be transformed.

        Returns
        -------
        t.Tuple[t.List[Node], t.List[Relationship]]
            A tuple containing a list of new nodes and a list of new relationships.
        """
        filtered = self.filter(kg)

        all_nodes = []
        all_relationships = []
        for node in filtered.nodes:
            nodes, relationships = await self.split(node)
            all_nodes.extend(nodes)
            all_relationships.extend(relationships)

        return all_nodes, all_relationships

    @abstractmethod
    async def split(self, node: Node) -> t.Tuple[t.List[Node], t.List[Relationship]]:
        """
        Abstract method to split a node into smaller chunks.

        Parameters
        ----------
        node : Node
            The node to be split.

        Returns
        -------
        t.Tuple[t.List[Node], t.List[Relationship]]
            A tuple containing a list of new nodes and a list of new relationships.
        """
        pass


class RelationshipBuilder(BaseGraphTransformations):
    """
    Abstract base class for building relationships in a KnowledgeGraph.

    Methods
    -------
    transform(kg: KnowledgeGraph) -> t.List[Relationship]
        Abstract method to transform the KnowledgeGraph by building relationships.
    """

    @abstractmethod
    async def transform(self, kg: KnowledgeGraph) -> t.List[Relationship]:
        """
        Transforms the KnowledgeGraph by building relationships.

        Parameters
        ----------
        kg : KnowledgeGraph
            The knowledge graph to be transformed.

        Returns
        -------
        t.List[Relationship]
            A list of new relationships.
        """
        pass


class Parallel:
    def __init__(self, *transformations: BaseGraphTransformations):
        self.transformations = list(transformations)


class Sequences:
    def __init__(self, *transformations: t.Union[BaseGraphTransformations, Parallel]):
        self.transformations = list(transformations)
