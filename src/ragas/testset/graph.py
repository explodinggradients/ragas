import json
import typing as t
import uuid
import random
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
        Finds indirect clusters of nodes in the knowledge graph based on a relationship condition.
        Here if A -> B -> C -> D, then A, B, C, and D form a cluster. If there's also a path A -> B -> C -> E,
        it will form a separate cluster.

        Parameters
        ----------
        relationship_condition : Callable[[Relationship], bool], optional
            A function that takes a Relationship and returns a boolean, by default lambda _: True

        Returns
        -------
        List[Set[Node]]
            A list of sets, where each set contains nodes that form a cluster.
        """
        clusters = []
        visited_paths = set()

        relationships = [
            rel for rel in self.relationships if relationship_condition(rel)
        ]

        def dfs(node: Node, cluster: t.Set[Node], depth: int, path: t.Tuple[Node, ...]):
            if depth >= depth_limit or path in visited_paths:
                return
            visited_paths.add(path)
            cluster.add(node)

            for rel in relationships:
                neighbor = None
                if rel.source == node and rel.target not in cluster:
                    neighbor = rel.target
                elif (
                    rel.bidirectional
                    and rel.target == node
                    and rel.source not in cluster
                ):
                    neighbor = rel.source

                if neighbor is not None:
                    dfs(neighbor, cluster.copy(), depth + 1, path + (neighbor,))

            # Add completed path-based cluster
            if len(cluster) > 1:
                clusters.append(cluster)

        for node in self.nodes:
            initial_cluster = set()
            dfs(node, initial_cluster, 0, (node,))

        # Remove duplicates by converting clusters to frozensets
        unique_clusters = [
            set(cluster) for cluster in set(frozenset(c) for c in clusters)
        ]

        return unique_clusters

    def find_n_indirect_clusters(
        self,
        n: int,
        relationship_condition: t.Callable[[Relationship], bool] = lambda _: True,
        depth_limit: int = 3,
    ) -> t.List[t.Set[Node]]:
        """
        Finds n indirect clusters of nodes in the knowledge graph based on a relationship condition.
        This is performant for large datasets as it only searches ~n paths and uses an adjacency index for lookups.
        Here if A -> B -> C -> D, then A, B, C, and D form a cluster. If there's also a path A -> B -> C -> E,
        it will form a separate cluster.  
        The end result is a list of n sets, where each set represents a full path from a starting node to a leaf node. 
        Paths are randomized in a way that maximizes variance by selecting n random starting nodes, grouping all their 
        paths and then iteratively selecting one item from each starting node group in a round-robin fashion until n 
        unique clusters are found. 
        To boost information breadth, we also lazily replace any subsets with found supersets when possible (non-exhaustive).
        
        Parameters
        ----------
        n : int
            Maximum number of clusters to return. The algorithm will use randomized path
            exploration to maximize variance.
        relationship_condition : Callable[[Relationship], bool], optional
            A function that takes a Relationship and returns a boolean, by default lambda _: True
        depth_limit : int, optional
            Maximum depth for path exploration, by default 3

        Returns
        -------
        List[Set[Node]]
            A list of sets, where each set contains nodes that form a cluster.
        """
        # Filter relationships once upfront
        filtered_relationships: list[Relationship] = [
            rel for rel in self.relationships if relationship_condition(rel)
        ]

        # Build adjacency list for faster neighbor lookup - optimized for large datasets
        adjacency_list: dict[Node, set[Node]] = {}
        for rel in filtered_relationships:
            # Lazy initialization since we only care about nodes with relationships
            if rel.source not in adjacency_list:
                adjacency_list[rel.source] = set()
            adjacency_list[rel.source].add(rel.target)

            if rel.bidirectional:
                if rel.target not in adjacency_list:
                    adjacency_list[rel.target] = set()
                adjacency_list[rel.target].add(rel.source)

        # Aggregate clusters for each start node
        start_node_clusters: dict[Node, set[frozenset[Node]]] = {}

        def dfs(node: Node, current_path: t.List[Node]):
            # Only check for cycles, depth limit is handled later
            if node in current_path:
                return

            current_path.append(node)

            # Check if we have any neighbors to explore
            neighbors = adjacency_list.get(node, set())
            
            # Filter out neighbors that are already in the current_path to handle cycles
            unvisited_neighbors = [n for n in neighbors if n not in current_path]

            # If this is a leaf node (no unvisited neighbors) or we've reached depth limit
            # and we have a valid path of at least 2 nodes, add it as a cluster
            is_leaf = len(unvisited_neighbors) == 0
            at_max_depth = len(current_path) == depth_limit
            start_node = current_path[0]
            if (is_leaf or at_max_depth) and len(current_path) > 1:
                # Lazy initialization of the set for this start node
                if start_node not in start_node_clusters:
                    start_node_clusters[start_node] = set()
                start_node_clusters[start_node].add(frozenset(current_path))
            else:
                for neighbor in unvisited_neighbors:
                    dfs(neighbor, current_path)

            # Backtrack by removing the current node from path
            current_path.pop()

        # Create a copy of nodes and shuffle them for random starting points
        # Use adjacency list since that has filtered out isolated nodes
        start_nodes = list(adjacency_list.keys())
        random.shuffle(start_nodes)
        # Get all the possible clusters for n start nodes
        for start_node in start_nodes[:n]:
            dfs(start_node, [])

        # Convert to list of sets for easier manipulation
        start_node_clusters_list: list[set[frozenset[Node]]] = list(start_node_clusters.values())
        
        # Iteratively pop from each start_node_clusters until we have n unique clusters
        unique_clusters = set()
        i = 0
        while len(unique_clusters) < n and start_node_clusters_list:
            # Cycle through the start node clusters
            current_index = i % len(start_node_clusters_list)
            
            # Pop a cluster and add it to unique_clusters
            cluster: frozenset[Node] = start_node_clusters_list[current_index].pop()
            
            # Remove any existing clusters that are subsets of this cluster
            existing_subsets = {c for c in unique_clusters if cluster.issuperset(c)}
            if existing_subsets:
                unique_clusters -= existing_subsets
            
            # Check if this cluster is a subset of any existing cluster
            if not any(cluster.issubset(c) for c in unique_clusters):
                # Add the cluster if it's not a subset of any existing cluster
                unique_clusters.add(cluster)
            
            # If this set is now empty, remove it
            if not start_node_clusters_list[current_index]:
                start_node_clusters_list.pop(current_index)
                # Don't increment i since we removed an element to account for shift
            else:
                i += 1
                
        return list(unique_clusters)

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
