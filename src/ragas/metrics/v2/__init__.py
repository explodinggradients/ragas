"""
V2 Metrics for Ragas.

This module provides metrics with:
    - Automatic validation and type safety
    - Full serialization support
    - Async-first design
    - Both class-based and decorator interfaces

Example:
    >>> from ragas.metrics.v2 import AnswerRelevancy
    >>> from ragas.llms import instructor_llm_factory
    >>> from ragas.embeddings import embedding_factory
    >>>
    >>> llm = instructor_llm_factory("openai", model="gpt-4o-mini")
    >>> embeddings = embedding_factory("openai", model="text-embedding-ada-002", interface="modern")
    >>>
    >>> metric = AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=3)
    >>> result = await metric.ascore(
    ...     user_input="What is Python?",
    ...     response="Python is a programming language"
    ... )
"""

from ragas.metrics.v2._answer_relevancy import AnswerRelevancy
from ragas.metrics.v2._rouge_score import RougeScore
from ragas.metrics.v2.base import V2BaseMetric
from ragas.metrics.v2.decorators import v2_metric

__all__ = [
    "V2BaseMetric",
    "AnswerRelevancy",
    "RougeScore",
    "v2_metric",
]
