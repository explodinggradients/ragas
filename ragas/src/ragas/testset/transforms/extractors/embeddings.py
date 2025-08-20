import typing as t
from dataclasses import dataclass, field

from ragas.embeddings import BaseRagasEmbedding, BaseRagasEmbeddings, embedding_factory
from ragas.testset.graph import Node
from ragas.testset.transforms.base import Extractor


@dataclass
class EmbeddingExtractor(Extractor):
    """
    A class for extracting embeddings from nodes in a knowledge graph.

    Attributes
    ----------
    property_name : str
        The name of the property to store the embedding
    embed_property_name : str
        The name of the property containing the text to embed
    embedding_model : BaseRagasEmbeddings or BaseRagasEmbedding
        The embedding model used for generating embeddings
    """

    property_name: str = "embedding"
    embed_property_name: str = "page_content"
    embedding_model: t.Union[BaseRagasEmbeddings, BaseRagasEmbedding] = field(
        default_factory=embedding_factory
    )

    async def extract(self, node: Node) -> t.Tuple[str, t.Any]:
        """
        Extracts the embedding for a given node.

        Raises
        ------
        ValueError
            If the property to be embedded is not a string.
        """
        text = node.get_property(self.embed_property_name)
        if not isinstance(text, str):
            raise ValueError(
                f"node.property('{self.embed_property_name}') must be a string, found '{type(text)}'"
            )
        # Handle both modern (BaseRagasEmbedding) and legacy (BaseRagasEmbeddings) interfaces
        if hasattr(self.embedding_model, "aembed_text"):
            # Modern interface (BaseRagasEmbedding)
            embedding = await self.embedding_model.aembed_text(text)  # type: ignore[attr-defined]
        else:
            # Legacy interface (BaseRagasEmbeddings)
            embedding = await self.embedding_model.embed_text(text)  # type: ignore[misc]
        return self.property_name, embedding
