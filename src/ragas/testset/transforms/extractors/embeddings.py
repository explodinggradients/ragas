import typing as t
from dataclasses import dataclass, field

from ragas.embeddings import BaseRagasEmbeddings, embedding_factory
from ragas.testset.graph import Node
from ragas.testset.transforms.base import Extractor


@dataclass
class EmbeddingExtractor(Extractor):
    property_name: str = "embedding"
    embed_property_name: str = "page_content"
    embedding_model: BaseRagasEmbeddings = field(default_factory=embedding_factory)

    async def extract(self, node: Node) -> t.Tuple[str, t.Any]:
        text = node.get_property(self.embed_property_name)
        if not isinstance(text, str):
            raise ValueError(
                f"node.property('{self.embed_property_name}') must be a string, found '{type(text)}'"
            )
        embedding = self.embedding_model.embed_query(text)
        return self.property_name, embedding
