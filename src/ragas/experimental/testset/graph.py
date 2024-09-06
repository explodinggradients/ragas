import typing as t
import uuid
from dataclasses import dataclass

from pydantic import BaseModel, Field


class Node(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    properties: dict = Field(default_factory=dict)
    type: str = ""

    # a simple repr
    def __repr__(self) -> str:
        return f"Node(id: {str(self.id)[:6]}, type: {self.type}, properties: {list(self.properties.keys())})"

    def add_property(self, key: str, value: t.Any):
        if key.lower() in self.properties:
            raise ValueError(f"Property {key} already exists")
        self.properties[key.lower()] = value


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
    nodes: t.List[Node] = Field(default_factory=list)
    relationships: t.List[Relationship] = Field(default_factory=list)

    def get_node_by_id(self, id: uuid.UUID) -> t.Optional[Node]:
        """
        Retrieve a node from the knowledge graph by its UUID.

        Args:
            id (uuid.UUID): The unique identifier of the node to retrieve.

        Returns:
            Optional[Node]: The node with the specified ID if found, None otherwise.
        """
        for node in self.nodes:
            if node.id == id:
                return node
        return None

    def get_relationship_by_id(self, id):
        for rel in self.relationships:
            if rel.id == id:
                return rel
