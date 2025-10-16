"""Context Precision metric v2 - Modern implementation with instructor LLMs."""

import typing as t

import numpy as np
from pydantic import BaseModel, Field

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult
from ragas.prompt.metrics.context_precision import context_precision_prompt

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM


class ContextPrecisionVerification(BaseModel):
    """Structured output for context precision verification."""

    reason: str = Field(..., description="Reason for the verdict")
    verdict: int = Field(..., description="Binary verdict: 1 if useful, 0 if not")


class ContextPrecision(BaseMetric):
    """
    Evaluate context precision using Average Precision metric.

    This metric evaluates whether all relevant items (contexts) are ranked higher
    by checking if each context was useful in arriving at the given answer.

    This implementation uses modern instructor LLMs with structured output.
    Only supports modern components - legacy wrappers are rejected with clear error messages.

    Usage:
        >>> from openai import AsyncOpenAI
        >>> from ragas.llms.base import instructor_llm_factory
        >>> from ragas.metrics.collections import ContextPrecision
        >>>
        >>> # Setup dependencies
        >>> client = AsyncOpenAI()
        >>> llm = instructor_llm_factory("openai", client=client, model="gpt-4o-mini")
        >>>
        >>> # Create metric instance
        >>> metric = ContextPrecision(llm=llm)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     user_input="What is the capital of France?",
        ...     retrieved_contexts=["Paris is the capital of France.", "London is in England."],
        ...     reference="Paris"
        ... )
        >>> print(f"Score: {result.value}")
        >>>
        >>> # Batch evaluation
        >>> results = await metric.abatch_score([
        ...     {"user_input": "Q1", "retrieved_contexts": ["C1", "C2"], "reference": "A1"},
        ...     {"user_input": "Q2", "retrieved_contexts": ["C1", "C2"], "reference": "A2"},
        ... ])

    Attributes:
        llm: Modern instructor-based LLM for verification
        name: The metric name
        allowed_values: Score range (0.0 to 1.0)
    """

    # Type hints for linter (attributes are set in __init__)
    llm: "InstructorBaseRagasLLM"

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        name: str = "context_precision",
        **kwargs,
    ):
        """Initialize ContextPrecision metric with required components."""
        # Set attributes explicitly before calling super()
        self.llm = llm

        # Call super() for validation
        super().__init__(name=name, **kwargs)

    async def ascore(
        self,
        user_input: str,
        retrieved_contexts: t.List[str],
        reference: str,
    ) -> MetricResult:
        """
        Calculate context precision score asynchronously.

        The metric evaluates each retrieved context to determine if it was useful
        for arriving at the reference answer, then calculates average precision.

        Args:
            user_input: The original question
            retrieved_contexts: List of retrieved context strings (in ranked order)
            reference: The reference answer to evaluate against

        Returns:
            MetricResult with average precision score (0.0-1.0)
        """
        # Handle edge cases
        if not retrieved_contexts:
            return MetricResult(value=0.0)

        if not reference or not user_input:
            return MetricResult(value=0.0)

        # Evaluate each context
        verdicts = []
        for context in retrieved_contexts:
            # Generate prompt for this context
            prompt = context_precision_prompt(
                question=user_input, context=context, answer=reference
            )

            # Get verification from LLM
            verification = await self.llm.agenerate(
                prompt, ContextPrecisionVerification
            )

            # Store binary verdict (1 if useful, 0 if not)
            verdicts.append(1 if verification.verdict else 0)

        # Calculate average precision
        score = self._calculate_average_precision(verdicts)

        return MetricResult(value=float(score))

    def _calculate_average_precision(self, verdict_list: t.List[int]) -> float:
        """
        Calculate average precision from list of binary verdicts.

        Average Precision formula:
        AP = (sum of (precision@k * relevance@k)) / (total relevant items)

        Where:
        - precision@k = (relevant items in top k) / k
        - relevance@k = 1 if item k is relevant, 0 otherwise

        Args:
            verdict_list: List of binary verdicts (1 for relevant, 0 for not)

        Returns:
            Average precision score (0.0-1.0), or nan if no relevant items
        """
        # Count total relevant items
        denominator = sum(verdict_list) + 1e-10

        # Calculate sum of precision at each relevant position
        numerator = sum(
            [
                (sum(verdict_list[: i + 1]) / (i + 1)) * verdict_list[i]
                for i in range(len(verdict_list))
            ]
        )

        score = numerator / denominator

        # Return nan if score is invalid
        if np.isnan(score):
            return np.nan

        return score
