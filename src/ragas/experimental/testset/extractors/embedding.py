import typing as t
from dataclasses import dataclass

from ragas.embeddings import BaseRagasEmbeddings, embedding_factory
from ragas.experimental.testset.graph import Node
from ragas.experimental.testset.graph_transforms import Extractor


@dataclass
class EmbeddingExtractor(Extractor):
    model: str = "text-embedding-3-small"
    property_name: str = "embedding"
    embedding_model: BaseRagasEmbeddings = embedding_factory(model=model)

    async def extract(self, node: Node) -> t.Tuple[str, t.Any]:
        text = node.get_property("page_content")
        if not isinstance(text, str):
            raise ValueError(
                f"node.property('page_content') must be a string, found '{type(text)}'"
            )
        embedding = self.embedding_model.embed_query(text)
        return self.property_name, embedding


embedding_extractor = EmbeddingExtractor()
