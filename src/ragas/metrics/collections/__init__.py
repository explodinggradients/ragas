"""Collections of metrics using modern component architecture."""

from ragas.metrics.collections._answer_relevancy import AnswerRelevancy
from ragas.metrics.collections._rouge_score import RougeScore
from ragas.metrics.collections.base import BaseMetric

__all__ = [
    "AnswerRelevancy",  # Class-based answer relevancy
    "RougeScore",  # Class-based rouge score
    "BaseMetric",  # Base class for creating new v2 metrics
]
