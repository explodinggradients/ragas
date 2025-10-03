"""V2 metrics using the decorator pattern for simpler implementation."""

from ragas.metrics.v2._answer_relevance import answer_relevancy
from ragas.metrics.v2._rouge_score import rouge_score

__all__ = [
    "answer_relevancy",
    "rouge_score",
]
