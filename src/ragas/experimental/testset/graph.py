import json
import typing as t
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class UUIDEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, uuid.UUID):
            return str(o)
        return super().default(o)


class NodeType(str, Enum):
    UNKNOWN = ""
    DOCUMENT = "document"
    CHUNK = "chunk"


class Node(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    properties: dict = Field(default_factory=dict)
    type: NodeType = NodeType.UNKNOWN

    # a simple repr
    def __repr__(self) -> str:
        return f"Node(id: {str(self.id)[:6]}, type: {self.type}, properties: {list(self.properties.keys())})"

    def __str__(self) -> str:
        return self.__repr__()

    def add_property(self, key: str, value: t.Any):
        if key.lower() in self.properties:
            raise ValueError(f"Property {key} already exists")
        self.properties[key.lower()] = value

    def get_property(self, key: str) -> t.Optional[t.Any]:
        return self.properties.get(key.lower(), None)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Node):
            return self.id == other.id
        return False


class Relationship(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    type: str
    source: Node
    target: Node
    bidirectional: bool = False
    properties: dict = Field(default_factory=dict)

    def get_property(self, key: str) -> t.Optional[t.Any]:
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


@dataclass
class KnowledgeGraph:
    nodes: t.List[Node] = field(default_factory=list)
    relationships: t.List[Relationship] = field(default_factory=list)

    def add(self, item: t.Union[Node, Relationship]):
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
        if isinstance(path, str):
            path = Path(path)

        data = {
            "nodes": [node.model_dump() for node in self.nodes],
            "relationships": [rel.model_dump() for rel in self.relationships],
        }
        with open(path, "w") as f:
            json.dump(data, f, cls=UUIDEncoder, indent=2)

    @classmethod
    def load(cls, path: t.Union[str, Path]) -> "KnowledgeGraph":
        if isinstance(path, str):
            path = Path(path)

        with open(path, "r") as f:
            data = json.load(f)

        nodes = [Node(**node_data) for node_data in data["nodes"]]
        relationships = [Relationship(**rel_data) for rel_data in data["relationships"]]

        kg = cls()
        kg.nodes.extend(nodes)
        kg.relationships.extend(relationships)
        return kg

    def __repr__(self) -> str:
        return f"KnowledgeGraph(nodes: {len(self.nodes)}, relationships: {len(self.relationships)})"

    def __str__(self) -> str:
        return self.__repr__()

    def find_clusters(
        self, relationship_condition: t.Callable[[Relationship], bool] = lambda _: True
    ) -> t.List[t.Set[Node]]:
        clusters = []
        visited = set()

        relationships = [
            rel for rel in self.relationships if relationship_condition(rel)
        ]

        def dfs(node: Node, cluster: t.Set[Node]):
            visited.add(node)
            cluster.add(node)
            for rel in relationships:
                if rel.source == node and rel.target not in visited:
                    dfs(rel.target, cluster)
                # if the relationship is bidirectional, we need to check the reverse
                elif (
                    rel.bidirectional
                    and rel.target == node
                    and rel.source not in visited
                ):
                    dfs(rel.source, cluster)

        for node in self.nodes:
            if node not in visited:
                cluster = set()
                dfs(node, cluster)
                if len(cluster) > 1:
                    clusters.append(cluster)

        return clusters
