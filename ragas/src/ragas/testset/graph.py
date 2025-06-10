import json
import typing as t
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_serializer


class UUIDEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, uuid.UUID):
            return str(o)
        return super().default(o)


class NodeType(str, Enum):
    """
    Enumeration of node types in the knowledge graph.

    Currently supported node types are: UNKNOWN, DOCUMENT, CHUNK
    """

    UNKNOWN = ""
    DOCUMENT = "document"
    CHUNK = "chunk"


class Node(BaseModel):
    """
    Represents a node in the knowledge graph.

    Attributes
    ----------
    id : uuid.UUID
        Unique identifier for the node.
    properties : dict
        Dictionary of properties associated with the node.
    type : NodeType
        Type of the node.

    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    properties: dict = Field(default_factory=dict)
    type: NodeType = NodeType.UNKNOWN

    def __repr__(self) -> str:
        return f"Node(id: {str(self.id)[:6]}, type: {self.type}, properties: {list(self.properties.keys())})"

    def __str__(self) -> str:
        return self.__repr__()

    def add_property(self, key: str, value: t.Any):
        """
        Adds a property to the node.

        Raises
        ------
        ValueError
            If the property already exists.
        """
        if key.lower() in self.properties:
            raise ValueError(f"Property {key} already exists")
        self.properties[key.lower()] = value

    def get_property(self, key: str) -> t.Optional[t.Any]:
        """
        Retrieves a property value by key.

        Notes
        -----
        The key is case-insensitive.
        """
        return self.properties.get(key.lower(), None)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Node):
            return self.id == other.id
        return False


class Relationship(BaseModel):
    """
    Represents a relationship between two nodes in a knowledge graph.

    Attributes
    ----------
    id : uuid.UUID, optional
        Unique identifier for the relationship. Defaults to a new UUID.
    type : str
        The type of the relationship.
    source : Node
        The source node of the relationship.
    target : Node
        The target node of the relationship.
    bidirectional : bool, optional
        Whether the relationship is bidirectional. Defaults to False.
    properties : dict, optional
        Dictionary of properties associated with the relationship. Defaults to an empty dict.

    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    type: str
    source: Node
    target: Node
    bidirectional: bool = False
    properties: dict = Field(default_factory=dict)

    def get_property(self, key: str) -> t.Optional[t.Any]:
        """
        Retrieves a property value by key. The key is case-insensitive.
        """
        return self.properties.get(key.lower(), None)

    def __repr__(self) -> str:
        return f"Relationship(Node(id: {str(self.source.id)[:6]}) {'<->' if self.bidirectional else '->'} Node(id: {str(self.target.id)[:6]}), type: {self.type}, properties: {list(self.properties.keys())})"

    def __str__(self) -> str:
        return self.__repr__()

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Relationship):
            return self.id == other.id
        return False

    @field_serializer("source", "target")
    def serialize_node(self, node: Node):
        return node.id


@dataclass
class KnowledgeGraph:
    """
    Represents a knowledge graph containing nodes and relationships.

    Attributes
    ----------
    nodes : List[Node]
        List of nodes in the knowledge graph.
    relationships : List[Relationship]
        List of relationships in the knowledge graph.
    """

    nodes: t.List[Node] = field(default_factory=list)
    relationships: t.List[Relationship] = field(default_factory=list)

    def add(self, item: t.Union[Node, Relationship]):
        """
        Adds a node or relationship to the knowledge graph.

        Raises
        ------
        ValueError
            If the item type is not Node or Relationship.
        """
        if isinstance(item, Node):
            self._add_node(item)
        elif isinstance(item, Relationship):
            self._add_relationship(item)
        else:
            raise ValueError(f"Invalid item type: {type(item)}")

    def _add_node(self, node: Node):
        self.nodes.append(node)

    def _add_relationship(self, relationship: Relationship):
        self.relationships.append(relationship)

    def save(self, path: t.Union[str, Path]):
        """Saves the knowledge graph to a JSON file.

        Parameters
        ----------
        path : Union[str, Path]
            Path where the JSON file should be saved.

        Notes
        -----
        The file is saved using UTF-8 encoding to ensure proper handling of Unicode characters
        across different platforms.
        """
        if isinstance(path, str):
            path = Path(path)

        data = {
            "nodes": [node.model_dump() for node in self.nodes],
            "relationships": [rel.model_dump() for rel in self.relationships],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, cls=UUIDEncoder, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: t.Union[str, Path]) -> "KnowledgeGraph":
        """Loads a knowledge graph from a path.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the JSON file containing the knowledge graph.

        Returns
        -------
        KnowledgeGraph
            The loaded knowledge graph.

        Notes
        -----
        The file is read using UTF-8 encoding to ensure proper handling of Unicode characters
        across different platforms.
        """
        if isinstance(path, str):
            path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        nodes = [Node(**node_data) for node_data in data["nodes"]]

        nodes_map = {str(node.id): node for node in nodes}
        relationships = [
            Relationship(
                id=rel_data["id"],
                type=rel_data["type"],
                source=nodes_map[rel_data["source"]],
                target=nodes_map[rel_data["target"]],
                bidirectional=rel_data["bidirectional"],
                properties=rel_data["properties"],
            )
            for rel_data in data["relationships"]
        ]

        kg = cls()
        kg.nodes.extend(nodes)
        kg.relationships.extend(relationships)
        return kg

    def __repr__(self) -> str:
        return f"KnowledgeGraph(nodes: {len(self.nodes)}, relationships: {len(self.relationships)})"

    def __str__(self) -> str:
        return self.__repr__()

    def find_indirect_clusters(
        self,
        relationship_condition: t.Callable[[Relationship], bool] = lambda _: True,
        depth_limit: int = 3,
    ) -> t.List[t.Set[Node]]:
        """
        Finds clusters (connected components) of nodes in the knowledge graph reachable within
        a limited number of hops, according to a relationship condition.

        A cluster is defined as a set of nodes such that each node in the cluster can be reached
        from any other node in the same cluster by traversing up to `depth_limit` relationships
        (edges), following only relationships that satisfy the given `relationship_condition`.

        For example, if A → B → C → D exists (and the relationships match the condition),
        then {A, B, C, D} will be grouped in the same cluster if all are within `depth_limit`
        of each other. Nodes that are not connected within this limit will be in separate clusters.

        Parameters
        ----------
        relationship_condition : Callable[[Relationship], bool], optional
            A function that takes a Relationship and returns a boolean indicating if it should be
            considered during clustering. By default, all relationships are considered.

        depth_limit : int, optional
            The maximum number of hops to use for clustering. Default is 3.

        Returns
        -------
        List[Set[Node]]
            A list of clusters: each cluster is a set of nodes, and no node appears in more than one cluster.
        """
        clusters = []
        assigned = set()
        relationships = [rel for rel in self.relationships if relationship_condition(rel)]

        # Build adjacency list
        adjacency = {node: set() for node in self.nodes}
        for rel in relationships:
            adjacency[rel.source].add(rel.target)
            if rel.bidirectional:
                adjacency[rel.target].add(rel.source)

        for node in self.nodes:
            if node not in assigned:
                # BFS for all nodes within depth_limit
                cluster = set([node])
                q = [(node, 0)]
                while q:
                    curr, depth = q.pop(0)
                    if depth == depth_limit:
                        continue
                    for neighbor in adjacency.get(curr, []):
                        if neighbor not in cluster:
                            cluster.add(neighbor)
                            q.append((neighbor, depth + 1))
                if len(cluster) > 1:
                    clusters.append(cluster)
                    assigned.update(cluster)
        return clusters

    def remove_node(
        self, node: Node, inplace: bool = True
    ) -> t.Optional["KnowledgeGraph"]:
        """
        Removes a node and its associated relationships from the knowledge graph.

        Parameters
        ----------
        node : Node
            The node to be removed from the knowledge graph.
        inplace : bool, optional
            If True, modifies the knowledge graph in place.
            If False, returns a modified copy with the node removed.

        Returns
        -------
        KnowledgeGraph or None
            Returns a modified copy of the knowledge graph if `inplace` is False.
            Returns None if `inplace` is True.

        Raises
        ------
        ValueError
            If the node is not present in the knowledge graph.
        """
        if node not in self.nodes:
            raise ValueError("Node is not present in the knowledge graph.")

        if inplace:
            # Modify the current instance
            self.nodes.remove(node)
            self.relationships = [
                rel
                for rel in self.relationships
                if rel.source != node and rel.target != node
            ]
        else:
            # Create a deep copy and modify it
            new_graph = deepcopy(self)
            new_graph.nodes.remove(node)
            new_graph.relationships = [
                rel
                for rel in new_graph.relationships
                if rel.source != node and rel.target != node
            ]
            return new_graph

    def find_two_nodes_single_rel(
        self, relationship_condition: t.Callable[[Relationship], bool] = lambda _: True
    ) -> t.List[t.Tuple[Node, Relationship, Node]]:
        """
        Finds nodes in the knowledge graph based on a relationship condition.
        (NodeA, NodeB, Rel) triples are considered as multi-hop nodes.

        Parameters
        ----------
        relationship_condition : Callable[[Relationship], bool], optional
            A function that takes a Relationship and returns a boolean, by default lambda _: True

        Returns
        -------
        List[Set[Node, Relationship, Node]]
            A list of sets, where each set contains two nodes and a relationship forming a multi-hop node.
        """

        relationships = [
            relationship
            for relationship in self.relationships
            if relationship_condition(relationship)
        ]

        triplets = set()

        for relationship in relationships:
            if relationship.source != relationship.target:
                node_a = relationship.source
                node_b = relationship.target
                # Ensure the smaller ID node is always first
                if node_a.id < node_b.id:
                    normalized_tuple = (node_a, relationship, node_b)
                else:
                    normalized_relationship = Relationship(
                        source=node_b,
                        target=node_a,
                        type=relationship.type,
                        properties=relationship.properties,
                    )
                    normalized_tuple = (node_b, normalized_relationship, node_a)

                triplets.add(normalized_tuple)

        return list(triplets)
