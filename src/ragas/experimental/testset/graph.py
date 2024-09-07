import typing as t
import uuid
from dataclasses import dataclass, field

from pydantic import BaseModel, Field


class Node(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    properties: dict = Field(default_factory=dict)
    type: str = ""

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


class Relationship(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    type: str
    properties: dict
    source: Node
    target: Node

    def __repr__(self) -> str:
        return f"Relationship(type: {self.type}, properties: {self.properties.keys()}, source: {self.source}, target: {self.target})"


@dataclass
class KnowledgeGraph:
    nodes: t.List[Node] = field(default_factory=list)
    relationships: t.List[Relationship] = field(default_factory=list)

    def add_node(self, node: Node):
        self.nodes.append(node)

    def add_relationship(self, relationship: Relationship):
        self.relationships.append(relationship)

    def get_node_by_id(self, id: uuid.UUID) -> t.Optional[Node]:
        for node in self.nodes:
            if node.id == id:
                return node
        return None

    def get_relationship_by_id(self, id):
        for rel in self.relationships:
            if rel.id == id:
                return rel
