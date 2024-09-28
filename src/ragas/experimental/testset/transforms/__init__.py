from .base import BaseGraphTransformation
from .engine import Parallel, Transforms, apply_transforms, rollback_transforms
from .extractors import (
    EmbeddingExtractor,
    HeadlinesExtractor,
    KeyphrasesExtractor,
    SummaryExtractor,
    TitleExtractor,
)
from .relationship_builders.cosine import CosineSimilarityBuilder
from .splitters import HeadlineSplitter


def default_transforms() -> Transforms:
    # define the transforms
    summary_extractor = SummaryExtractor()
    keyphrase_extractor = KeyphrasesExtractor()
    title_extractor = TitleExtractor()
    headline_extractor = HeadlinesExtractor()
    embedding_extractor = EmbeddingExtractor()
    headline_splitter = HeadlineSplitter()
    cosine_sim_builder = CosineSimilarityBuilder(threshold=0.8)

    # specify the transforms and their order to be applied
    transforms = [
        headline_extractor,
        headline_splitter,
        Parallel(
            embedding_extractor, summary_extractor, keyphrase_extractor, title_extractor
        ),
        cosine_sim_builder,
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
