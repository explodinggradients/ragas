"""Example of creating a new v2 metric using V2BaseMetric."""

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult


class ExampleMetric(BaseMetric):
    """
    Example metric showing how easy it is to create new metrics.

    This metric inherits all the validation and base functionality from BaseMetric:
    - Batch processing capabilities
    - Type safety
    - Async-first design

    Usage:
        >>> metric = ExampleMetric()
        >>> result = await metric.ascore(user_input="test", response="test")
    """

    def __init__(self, name: str = "example_metric", **kwargs):
        """Initialize the example metric."""
        super().__init__(name=name, **kwargs)

    async def ascore(self, user_input: str, response: str) -> MetricResult:
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
