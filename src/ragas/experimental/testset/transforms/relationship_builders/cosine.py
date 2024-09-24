import typing as t
from dataclasses import dataclass

import numpy as np

from ragas.experimental.testset.graph import KnowledgeGraph, Relationship
from ragas.experimental.testset.transforms.base import RelationshipBuilder


@dataclass
class JaccardSimilarityBuilder(RelationshipBuilder):
    # TODO: Implement
    pass


@dataclass
class CosineSimilarityBuilder(RelationshipBuilder):
    attribute: t.Optional[str] = None
    threshold: float = 0.9

    def _find_similar_embedding_pairs(
        self, embeddings: np.ndarray, threshold: float
    ) -> t.List[t.Tuple[int, int, float]]:
        # Normalize the embeddings
        normalized = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]

        # Calculate cosine similarity matrix
        similarity_matrix = np.dot(normalized, normalized.T)
        # Find pairs with similarity >= threshold
        similar_pairs = np.argwhere(similarity_matrix >= threshold)

        # Filter out self-comparisons and duplicate pairs
        return [
            (pair[0], pair[1], similarity_matrix[pair[0], pair[1]])
            for pair in similar_pairs
            if pair[0] < pair[1]
        ]

    async def transform(self, kg: KnowledgeGraph) -> t.List[Relationship]:
        if self.attribute is None:
            self.attribute = "embedding"

        embeddings = []
        for node in kg.nodes:
            embedding = node.get_property(self.attribute)
            if embedding is None:
                raise ValueError(f"Node {node.id} has no {self.attribute}")
            embeddings.append(embedding)

        similar_pairs = self._find_similar_embedding_pairs(
            np.array(embeddings), self.threshold
        )

        return [
            Relationship(
                source=kg.nodes[i],
                target=kg.nodes[j],
                type="cosine_similarity",
                properties={"cosine_similarity": similarity_float},
                bidirectional=True,
            )
            for i, j, similarity_float in similar_pairs
        ]
