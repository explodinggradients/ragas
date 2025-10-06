"""Base class for v2 metrics."""

import asyncio
import typing as t
from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ragas.metrics.result import MetricResult

if t.TYPE_CHECKING:
    pass


class V2BaseMetric(BaseModel, ABC):
    """
    Base class for v2 metrics.

    Attributes:
        name: The metric name
        allowed_values: Score range for numeric validation (tuple of min, max)

    Example:
        >>> class MyMetric(V2BaseMetric):
        ...     name: str = "my_metric"
        ...     llm: InstructorBaseRagasLLM
        ...     threshold: float = 0.5
        ...
        ...     async def _ascore_impl(self, user_input: str, response: str):
        ...         return MetricResult(value=0.95)
        >>>
        >>> metric = MyMetric(llm=my_llm, threshold=0.8)
        >>> result = await metric.ascore(user_input="Q", response="A")
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
    )

    # Core fields
    name: str = Field(..., description="The name of this metric")
    allowed_values: t.Tuple[float, float] = Field(
        default=(0.0, 1.0),
        description="Valid range for metric scores as (min, max)",
    )

    @field_validator("allowed_values")
    @classmethod
    def validate_allowed_values(cls, v: t.Tuple[float, float]) -> t.Tuple[float, float]:
        """Validate that allowed_values is a valid range tuple."""
        if not isinstance(v, tuple) or len(v) != 2:
            raise ValueError("allowed_values must be a tuple of (min, max)")

        min_val, max_val = v
        if not isinstance(min_val, (int, float)) or not isinstance(
            max_val, (int, float)
        ):
            raise ValueError("allowed_values must contain numeric values")

        if min_val >= max_val:
            raise ValueError(
                f"allowed_values min ({min_val}) must be less than max ({max_val})"
            )

        return v

    async def ascore(self, **kwargs) -> MetricResult:
        """
        Calculate metric score asynchronously.

        Args:
            **kwargs: Metric-specific input parameters

        Returns:
            MetricResult with validated score
        """
        # Call the subclass implementation
        result = await self._ascore_impl(**kwargs)

        # Validate the result value
        if result.value is not None:
            validation_error = self._validate_result_value(result.value)
            if validation_error:
                return MetricResult(value=None, reason=validation_error)

        return result

    @abstractmethod
    async def _ascore_impl(self, **kwargs) -> MetricResult:
        """
        Implement metric-specific scoring logic.

        Args:
            **kwargs: Metric-specific input parameters

        Returns:
            MetricResult with score and optional reasoning
        """
        ...

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

        Example:
            >>> results = await metric.abatch_score([
            ...     {"user_input": "Q1", "response": "A1"},
            ...     {"user_input": "Q2", "response": "A2"},
            ... ])
            >>> for r in results:
            ...     print(r.value)
        """
        # Create async tasks for concurrent execution
        async_tasks = [self.ascore(**input_dict) for input_dict in inputs]

        # Run all tasks concurrently
        return await asyncio.gather(*async_tasks)

    def score(self, **kwargs) -> MetricResult:
        """
        Synchronous scoring method.

        Args:
            **kwargs: Metric-specific input parameters

        Returns:
            MetricResult object

        Raises:
            RuntimeError: If called from within an async context

        Example:
            >>> result = metric.score(user_input="Q", response="A")
        """
        try:
            # Check if we're already in an async context
            asyncio.get_running_loop()
            # If we get here, there's already a running loop
            raise RuntimeError(
                "Cannot call sync score() from an async context. "
                "Use ascore() instead, or use ragas.async_utils.run() if needed."
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
        Synchronous batch scoring.

        Args:
            inputs: List of input dictionaries for scoring

        Returns:
            List of MetricResult objects

        Raises:
            RuntimeError: If called from within an async context
        """
        try:
            # Check if we're already in an async context
            asyncio.get_running_loop()
            # If we get here, there's already a running loop
            raise RuntimeError(
                "Cannot call sync batch_score() from an async context. "
                "Use abatch_score() instead, or use ragas.async_utils.run() if needed."
            )
        except RuntimeError as e:
            if "Use abatch_score() instead" in str(e):
                raise  # Re-raise our custom error
            # No running loop found, safe to use asyncio.run()
            return asyncio.run(self.abatch_score(inputs))

    def _validate_result_value(self, result_value: t.Any) -> t.Optional[str]:
        """
        Validate that result value is within the numeric range.

        Args:
            result_value: The value to validate

        Returns:
            Error message if validation fails, None if valid
        """
        if not isinstance(result_value, (int, float)):
            return (
                f"Metric {self.name} returned '{result_value}' "
                f"but expected a numeric value"
            )

        min_val, max_val = self.allowed_values
        if not (min_val <= result_value <= max_val):
            return (
                f"Metric {self.name} returned {result_value} "
                f"but expected value in range {self.allowed_values}"
            )

        return None

    def save_config(self, exclude_components: bool = True) -> t.Dict[str, t.Any]:
        """
        Save metric configuration to a dictionary.

        Args:
            exclude_components: If True, excludes LLM/embeddings components

        Returns:
            Dictionary containing the metric configuration

        Example:
            >>> config = metric.save_config(exclude_components=True)
            >>> new_metric = AnswerRelevancy(**config, llm=new_llm, embeddings=new_emb)
        """
        exclude_set = set()
        if exclude_components:
            for field_name in self.__class__.model_fields:
                field_value = getattr(self, field_name, None)
                if field_value is not None and hasattr(field_value, "agenerate"):
                    exclude_set.add(field_name)
                elif field_value is not None and hasattr(field_value, "embed_text"):
                    exclude_set.add(field_name)

        return self.model_dump(exclude=exclude_set)

    def __repr__(self) -> str:
        """Return a clean string representation of the metric."""
        fields = []
        # Use self.__class__.model_fields instead of self.model_fields
        for field_name in self.__class__.model_fields.keys():
            value = getattr(self, field_name, None)

            # Special handling for different types
            if field_name == "name":
                fields.append(f"name='{value}'")
            elif field_name == "allowed_values":
                fields.append(f"allowed_values={value}")
            elif hasattr(value, "__class__") and hasattr(value.__class__, "__name__"):
                # For complex objects, just show the type
                type_name = value.__class__.__name__
                fields.append(f"{field_name}={type_name}(...)")
            elif value is not None and not isinstance(value, type):
                fields.append(f"{field_name}={value!r}")

        return f"{self.__class__.__name__}({', '.join(fields)})"
