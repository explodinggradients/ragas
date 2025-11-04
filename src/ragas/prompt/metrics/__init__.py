"""Metric-specific prompts for Ragas evaluation metrics."""

from ragas.prompt.metrics.answer_correctness import correctness_classifier_prompt
from ragas.prompt.metrics.answer_relevance import answer_relevancy_prompt
from ragas.prompt.metrics.common import nli_statement_prompt, statement_generator_prompt
from ragas.prompt.metrics.context_precision import (
    context_precision_prompt,
    context_precision_with_reference_prompt,
    context_precision_without_reference_prompt,
)

__all__ = [
    "answer_relevancy_prompt",
    "context_precision_prompt",
    "context_precision_with_reference_prompt",
    "context_precision_without_reference_prompt",
    "correctness_classifier_prompt",
    "nli_statement_prompt",
    "statement_generator_prompt",
]
