import json
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

from langchain_core.documents import Document as LCDocument
from ragas_experimental.testset.graph import Node


@dataclass
class Extractor(ABC):
    attribute: str = "page_content"

    @abstractmethod
    async def aextract_text(self, text: str) -> t.Any:
        raise NotImplementedError("aextract_text() is not implemented for Extractor")

    @abstractmethod
    def extract_text(self, text: str) -> t.Any:
        raise NotImplementedError("extract() is not implemented for Extractor")

    async def aextract(self, node: t.Union[Node, LCDocument]) -> t.Any:
        if isinstance(node, Node):
            if self.attribute in node.properties:
                return await self.aextract_text(node.properties[self.attribute])
            elif self.attribute in node.properties["metadata"]:
                return await self.aextract_text(
                    json.dumps(node.properties["metadata"][self.attribute])
                )
            else:
                raise ValueError(f"Attribute {self.attribute} not found in node")
        elif isinstance(node, LCDocument):
            if self.attribute == "page_content":
                return await self.aextract_text(node.page_content)
            elif self.attribute in node.metadata:
                return await self.aextract_text(
                    json.dumps(node.metadata[self.attribute])
                )
            else:
                raise ValueError(f"Attribute {self.attribute} not found in node")

    def extract(self, node: t.Union[Node, LCDocument]) -> t.Any:
        if isinstance(node, Node):
            if self.attribute in node.properties:
                return self.extract_text(node.properties[self.attribute])
            elif self.attribute in node.properties["metadata"]:
                return self.extract_text(
                    json.dumps(node.properties["metadata"][self.attribute])
                )
            else:
                raise ValueError(f"Attribute {self.attribute} not found in node")
        elif isinstance(node, LCDocument):
            if self.attribute == "page_content":
                return self.extract_text(node.page_content)
            elif self.attribute in node.metadata:
                return self.extract_text(json.dumps(node.metadata[self.attribute]))
            else:
                raise ValueError(f"Attribute {self.attribute} not found in node")

    def merge_extractors(self, *extractors) -> t.List["Extractor"]:
        raise NotImplementedError("merge_extractors() is not implemented for Extractor")


@dataclass
class Regex:
    name: str
    pattern: str

    def __call__(self):
        # Ensure the pattern is a raw string
        if not isinstance(self.pattern, str):
            raise TypeError("Pattern must be a string.")

        if not isinstance(self.name, str):
            raise TypeError("Group name must be a string.")

        # Add the named group to the pattern
        return (
            f"(?P<{self.name}>{self.pattern})"
            if self.name != "merged_extractor"
            else self.pattern
        )
