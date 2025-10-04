"""Base class for v2 metrics with modern component validation."""

import asyncio
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ragas.metrics.result import MetricResult

if t.TYPE_CHECKING:
    pass


@dataclass
class V2BaseMetric(ABC):
    """
    Base class for v2 metrics with modern component validation.

    This class provides:
    - Automatic validation of modern LLM and embedding components (when defined by subclass)
    - Rejection of legacy wrappers with helpful error messages
    - Numeric validation with configurable ranges
    - Consistent error handling and type safety
    - Pure async design with concurrent batch processing

    Attributes:
        name: The metric name
        allowed_values: Score range for numeric validation (tuple of min, max)

    Note: Subclasses define llm and/or embeddings fields only if they need them.
    """

    name: str
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
        Async scoring method with automatic result validation.

        Components are guaranteed to be validated and non-None after __post_init__.
        Results are automatically validated against allowed_values range.
        """
        # Call the subclass implementation
        result = await self._ascore_impl(**kwargs)

        # Validate the result value
        if result.value is not None:
            validation_error = self.validate_result_value(result.value)
            if validation_error:
                return MetricResult(value=None, reason=validation_error)

        return result

    @abstractmethod
    async def _ascore_impl(self, **kwargs) -> MetricResult:
        """
        Metric-specific async scoring implementation.

        Subclasses should override this method instead of ascore().
        Components are guaranteed to be validated and non-None.
        Results will be automatically validated by the base class.
        """
        pass

    async def abatch_score(
        self,
        inputs: t.List[t.Dict[str, t.Any]],
    ) -> t.List[MetricResult]:
        """
        Async batch scoring with concurrent execution.

        Args:
            inputs: List of input dictionaries for scoring

        Returns:
            List of MetricResult objects
        """
        async_tasks = []
        for input_dict in inputs:
            # Process input asynchronously
            async_tasks.append(self.ascore(**input_dict))

        # Run all tasks concurrently and return results
        return await asyncio.gather(*async_tasks)

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

    def validate_result_value(self, result_value: t.Any) -> t.Optional[str]:
        """Validate that result value is within the numeric range."""
        if not isinstance(result_value, (int, float)):
            return f"Metric {self.name} returned '{result_value}' but expected a numeric value"

        min_val, max_val = self.allowed_values
        if not (min_val <= result_value <= max_val):
            return f"Metric {self.name} returned {result_value} but expected value in range {self.allowed_values}"

        return None

    def _validate_llm(self):
        """Validate that a modern InstructorLLM is provided."""
        llm = getattr(self, "llm", None)
        if llm is None:
            raise ValueError(f"{self.__class__.__name__} requires an llm parameter")

        if type(llm).__name__ != "InstructorLLM":
            raise ValueError(
                f"V2 metrics only support modern InstructorLLM. Found: {type(llm).__name__}. "
                f"Use: instructor_llm_factory('openai', model='gpt-4o-mini', client=openai_client)"
            )

    def _validate_embeddings(self):
        """Validate that modern embeddings are provided."""
        embeddings = getattr(self, "embeddings", None)
        if embeddings is None:
            raise ValueError(
                f"{self.__class__.__name__} requires an embeddings parameter"
            )

        if type(embeddings).__name__ == "LangchainEmbeddingsWrapper":
            raise ValueError(
                "V2 metrics only support modern embeddings. Legacy LangchainEmbeddingsWrapper is not supported. "
                "Use: embedding_factory('openai', model='text-embedding-ada-002', client=openai_client, interface='modern')"
            )
