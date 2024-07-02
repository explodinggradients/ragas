import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ragas_experimental.testset.graph import Node


@dataclass
class Similarity(ABC):
    name: str
    attribute1: str
    attribute2: str

    def get_attribute(self, node: Node, attribute: str):
        if attribute == "page_content":
            return node.properties["page_content"]
        elif attribute in node.properties["metadata"]:
            return node.properties["metadata"][attribute]
        else:
            raise ValueError(f"Attribute {attribute} not found in node")

    @abstractmethod
    def extract(self, x_nodes: t.List[Node], y_nodes: t.List[Node]) -> t.Any:
        pass
