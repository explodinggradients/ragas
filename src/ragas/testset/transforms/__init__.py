from .base import BaseGraphTransformation
from .engine import Parallel, Transforms, apply_transforms, rollback_transforms
from .extractors import (
    EmbeddingExtractor,
    HeadlinesExtractor,
    KeyphrasesExtractor,
    SummaryExtractor,
    TitleExtractor,
)
from .relationship_builders.cosine import (
    CosineSimilarityBuilder,
    SummaryCosineSimilarityBuilder,
)
from .splitters import HeadlineSplitter


def default_transforms() -> Transforms:
    from ragas.testset.graph import NodeType

    # define the transforms
    summary_extractor = SummaryExtractor()
    keyphrase_extractor = KeyphrasesExtractor()
    title_extractor = TitleExtractor()
    headline_extractor = HeadlinesExtractor()
    embedding_extractor = EmbeddingExtractor()
    headline_splitter = HeadlineSplitter()
    cosine_sim_builder = CosineSimilarityBuilder(threshold=0.8)
    summary_embedder = EmbeddingExtractor(
        name="summary_embedder",
        property_name="summary_embedding",
        embed_property_name="summary",
        filter_nodes=lambda node: True if node.type == NodeType.DOCUMENT else False,
    )
    summary_cosine_sim_builder = SummaryCosineSimilarityBuilder(threshold=0.6)

    # specify the transforms and their order to be applied
    transforms = [
        Parallel(summary_extractor, headline_extractor),
        summary_embedder,
        headline_splitter,
        Parallel(embedding_extractor, keyphrase_extractor, title_extractor),
        cosine_sim_builder,
        summary_cosine_sim_builder,
    ]
    return transforms


__all__ = [
    "BaseGraphTransformation",
    "Parallel",
    "Transforms",
    "apply_transforms",
    "rollback_transforms",
    "default_transforms",
]
