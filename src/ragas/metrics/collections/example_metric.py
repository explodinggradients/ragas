"""Example of creating a new v2 metric using V2BaseMetric."""

import typing as t
from dataclasses import dataclass

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult

if t.TYPE_CHECKING:
    from ragas.embeddings.base import BaseRagasEmbedding
    from ragas.llms.base import InstructorBaseRagasLLM


@dataclass(kw_only=True)
class ExampleMetric(BaseMetric):
    """
    Example metric showing how easy it is to create new metrics.

    This metric inherits all the validation and base functionality from V2BaseMetric:
    - Automatic LLM and embedding validation
    - Numeric validation with configurable ranges
    - Batch processing capabilities
    - Type safety
    - Async-first design

    Usage:
        >>> metric = ExampleMetric(llm=modern_llm, embeddings=modern_embeddings)
        >>> result = await metric.ascore(user_input="test", response="test")
    """

    name: str = "example_metric"
    llm: "InstructorBaseRagasLLM"
    embeddings: "BaseRagasEmbedding"

    async def _ascore_impl(self, user_input: str, response: str) -> MetricResult:
        """
        Calculate example score asynchronously.

        Components are guaranteed to be validated and non-None by the base class.

        Args:
            user_input: The original question
            response: The response to evaluate

        Returns:
            MetricResult with example score
        """
        # Example logic - just return a simple score based on response length
        # In a real metric, you'd use self.llm and self.embeddings
        score = min(len(response) / 100.0, 1.0)  # Cap at 1.0

        return MetricResult(value=float(score))


# This is how simple it is to create a new v2 metric!
# The base class handles all the validation, type safety, and batch processing.
