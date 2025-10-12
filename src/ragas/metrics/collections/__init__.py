"""Collections of metrics using modern component architecture."""

from ragas.metrics.collections._answer_relevancy import AnswerRelevancy
from ragas.metrics.collections._answer_similarity import AnswerSimilarity
from ragas.metrics.collections._bleu_score import BleuScore
from ragas.metrics.collections._rouge_score import RougeScore
from ragas.metrics.collections.base import BaseMetric

__all__ = [
    "BaseMetric",  # Base class
    "AnswerRelevancy",
    "AnswerSimilarity",
    "BleuScore",
    "RougeScore",
]
