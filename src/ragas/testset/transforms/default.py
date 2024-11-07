from __future__ import annotations

import typing as t

from ragas.testset.graph import NodeType
from ragas.testset.transforms.extractors import (
    EmbeddingExtractor,
    HeadlinesExtractor,
    SummaryExtractor,
)
from ragas.testset.transforms.extractors.llm_based import NERExtractor, ThemesExtractor
from ragas.testset.transforms.relationship_builders import (
    CosineSimilarityBuilder,
    OverlapScoreBuilder,
)
from ragas.testset.transforms.splitters import HeadlineSplitter
from ragas.utils import num_tokens_from_string

from .engine import Parallel

if t.TYPE_CHECKING:
    from ragas.embeddings.base import BaseRagasEmbeddings
    from ragas.llms.base import BaseRagasLLM

    from .engine import Transforms


def default_transforms(
    llm: BaseRagasLLM,
    embedding_model: BaseRagasEmbeddings,
) -> Transforms:
    """
    Creates and returns a default set of transforms for processing a knowledge graph.

    This function defines a series of transformation steps to be applied to a
    knowledge graph, including extracting summaries, keyphrases, titles,
    headlines, and embeddings, as well as building similarity relationships
    between nodes.



    Returns
    -------
    Transforms
        A list of transformation steps to be applied to the knowledge graph.

    """

    headline_extractor = HeadlinesExtractor(llm=llm)
    splitter = HeadlineSplitter(min_tokens=500)

    def summary_filter(node):
        return (
            node.type == NodeType.DOCUMENT
            and num_tokens_from_string(node.properties["page_content"]) > 500
        )

    summary_extractor = SummaryExtractor(
        llm=llm, filter_nodes=lambda node: summary_filter(node)
    )

    theme_extractor = ThemesExtractor(llm=llm)
    ner_extractor = NERExtractor(
        llm=llm, filter_nodes=lambda node: node.type == NodeType.CHUNK
    )

    summary_emb_extractor = EmbeddingExtractor(
        embedding_model=embedding_model,
        property_name="summary_embedding",
        embed_property_name="summary",
        filter_nodes=lambda node: summary_filter(node),
    )

    cosine_sim_builder = CosineSimilarityBuilder(
        property_name="summary_embedding",
        new_property_name="summary_similarity",
        threshold=0.7,
        filter_nodes=lambda node: summary_filter(node),
    )

    ner_overlap_sim = OverlapScoreBuilder(
        threshold=0.01, filter_nodes=lambda node: node.type == NodeType.CHUNK
    )

    transforms = [
        headline_extractor,
        splitter,
        Parallel(summary_extractor, theme_extractor, ner_extractor),
        summary_emb_extractor,
        Parallel(cosine_sim_builder, ner_overlap_sim),
    ]

    return transforms
