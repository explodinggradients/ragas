"""Base class for collections metrics with modern component validation."""

import asyncio
import typing as t
from dataclasses import dataclass

from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.base import SimpleBaseMetric
from ragas.metrics.result import MetricResult
from ragas.metrics.validators import NumericValidator


@dataclass
class BaseMetric(SimpleBaseMetric, NumericValidator):
    """
    Base class for metrics collections with modern component validation.

    This class inherits from SimpleBaseMetric and NumericValidator to provide:
    - All the base metric functionality (ascore, abatch_score, score, batch_score)
    - Numeric validation with configurable ranges
    - Modern LLM and embedding component validation (when defined by subclass)
    - Rejection of legacy wrappers with helpful error messages
    - Consistent error handling and type safety

    Attributes:
        name: The metric name
        allowed_values: Score range for numeric validation (tuple of min, max)

    Note: Subclasses define llm and/or embeddings fields only if they need them.
    The base classes handle all the core metric functionality - we just add modern component validation.
    """

    name: str = "base_metric"
    allowed_values: t.Tuple[float, float] = (0.0, 1.0)

    def __post_init__(self):
        """Validate components only if the metric defines them."""
        # Check if this specific metric class defines these fields
        if "llm" in self.__dataclass_fields__:
            self._validate_llm()
        if "embeddings" in self.__dataclass_fields__:
            self._validate_embeddings()

    async def ascore(self, **kwargs) -> MetricResult:
        """
        Default async scoring method - subclasses should override this.

        This base implementation just returns a placeholder result.
        Subclasses should override this method with their specific logic.

        The base class handles component validation in __post_init__.
        """
        return MetricResult(
            value=0.0, reason="Base metric placeholder - override ascore() in subclass"
        )

    def score(self, **kwargs) -> MetricResult:
        """
        Synchronous scoring method that wraps ascore().

        This is a convenience method for backward compatibility and sync usage.
        For better performance, prefer using ascore() directly in async contexts.

        Returns:
            MetricResult object
        """
        try:
            # Check if we're already in an async context
            asyncio.get_running_loop()
            # If we get here, there's already a running loop
            raise RuntimeError(
                "Cannot call sync score() from an async context. Use ascore() instead."
            )
        except RuntimeError as e:
            if "Use ascore() instead" in str(e):
                raise  # Re-raise our custom error
            # No running loop found, safe to use asyncio.run()
            return asyncio.run(self.ascore(**kwargs))

    def batch_score(
        self,
        inputs: t.List[t.Dict[str, t.Any]],
    ) -> t.List[MetricResult]:
        """
        Synchronous batch scoring that wraps abatch_score().

        This is a convenience method for backward compatibility and sync usage.
        For better performance, prefer using abatch_score() directly in async contexts.

        Args:
            inputs: List of input dictionaries for scoring

        Returns:
            List of MetricResult objects
        """
        try:
            # Check if we're already in an async context
            asyncio.get_running_loop()
            # If we get here, there's already a running loop
            raise RuntimeError(
                "Cannot call sync batch_score() from an async context. Use abatch_score() instead."
            )
        except RuntimeError as e:
            if "Use abatch_score() instead" in str(e):
                raise  # Re-raise our custom error
            # No running loop found, safe to use asyncio.run()
            return asyncio.run(self.abatch_score(inputs))

    def _validate_llm(self):
        """Validate that a modern InstructorLLM is provided."""
        llm = getattr(self, "llm", None)

        if not isinstance(llm, InstructorBaseRagasLLM):
            raise ValueError(
                f"Collections metrics only support modern InstructorLLM. Found: {type(llm).__name__}. "
                f"Use: instructor_llm_factory('openai', model='gpt-4o-mini', client=openai_client)"
            )

    def _validate_embeddings(self):
        """Validate that modern embeddings are provided."""
        embeddings = getattr(self, "embeddings", None)

        if not isinstance(embeddings, BaseRagasEmbedding):
            raise ValueError(
                f"Collections metrics only support modern embeddings. Found: {type(embeddings).__name__}. "
                f"Use: embedding_factory('openai', model='text-embedding-ada-002', client=openai_client, interface='modern')"
            )
