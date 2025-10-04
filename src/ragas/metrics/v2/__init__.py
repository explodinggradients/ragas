"""V2 metrics using modern component architecture."""

from ragas.metrics.v2._answer_relevancy import AnswerRelevancy
from ragas.metrics.v2._rouge_score import RougeScore
from ragas.metrics.v2.base import V2BaseMetric

__all__ = [
    "AnswerRelevancy",  # Class-based answer relevancy
    "RougeScore",  # Class-based rouge score
    "V2BaseMetric",  # Base class for creating new v2 metrics
]
