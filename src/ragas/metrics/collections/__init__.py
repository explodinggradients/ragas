"""Collections of metrics using modern component architecture."""

from ragas.metrics.collections._answer_correctness import AnswerCorrectness
from ragas.metrics.collections._answer_relevancy import AnswerRelevancy
from ragas.metrics.collections._answer_similarity import AnswerSimilarity
from ragas.metrics.collections._aspect_critic import (
    AspectCritic,
    coherence,
    conciseness,
    correctness,
    harmfulness,
    maliciousness,
)
from ragas.metrics.collections._bleu_score import BleuScore
from ragas.metrics.collections._context_entity_recall import ContextEntityRecall
from ragas.metrics.collections._rouge_score import RougeScore
from ragas.metrics.collections._semantic_similarity import SemanticSimilarity
from ragas.metrics.collections._simple_criteria import SimpleCriteria
from ragas.metrics.collections._string import (
    DistanceMeasure,
    ExactMatch,
    NonLLMStringSimilarity,
    StringPresence,
)
from ragas.metrics.collections._summary_score import SummaryScore
from ragas.metrics.collections.base import BaseMetric

__all__ = [
    "BaseMetric",  # Base class
    "AnswerCorrectness",
    "AnswerRelevancy",
    "AnswerSimilarity",
    "AspectCritic",
    "BleuScore",
    "ContextEntityRecall",
    "DistanceMeasure",
    "ExactMatch",
    "NonLLMStringSimilarity",
    "RougeScore",
    "SemanticSimilarity",
    "SimpleCriteria",
    "StringPresence",
    "SummaryScore",
    # AspectCritic helper functions
    "coherence",
    "conciseness",
    "correctness",
    "harmfulness",
    "maliciousness",
]
