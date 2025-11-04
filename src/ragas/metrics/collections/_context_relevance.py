"""Context Relevance metric v2 - Modern implementation with dual-judge evaluation."""

import typing as t
from typing import List

import numpy as np
from pydantic import BaseModel

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult
from ragas.prompt.metrics.context_relevance import (
    context_relevance_judge1_prompt,
    context_relevance_judge2_prompt,
)

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM


class RelevanceRating(BaseModel):
    """Structured output for relevance rating."""

    rating: int


class ContextRelevance(BaseMetric):
    """
    Modern v2 implementation of context relevance evaluation.

    Evaluates whether the retrieved contexts are pertinent to the user input
    using a dual-judge system. This metric averages two distinct judge prompts
    to ensure robust evaluation.

    The metric uses NVIDIA's proven dual-judge approach:
    1. Judge 1: Direct context relevance evaluation
    2. Judge 2: Alternative perspective for fairness
    3. Average both judges for final score

    Rating scale: 0 (not relevant), 1 (partially relevant), 2 (fully relevant)
    Final score: Average of both judges converted to 0.0-1.0 scale

    Usage:
        >>> import instructor
        >>> from openai import AsyncOpenAI
        >>> from ragas.llms.base import instructor_llm_factory
        >>> from ragas.metrics.collections import ContextRelevance
        >>>
        >>> # Setup dependencies
        >>> client = AsyncOpenAI()
        >>> llm = instructor_llm_factory("openai", client=client, model="gpt-4o")
        >>>
        >>> # Create metric instance
        >>> metric = ContextRelevance(llm=llm)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     user_input="When was Einstein born?",
        ...     retrieved_contexts=["Albert Einstein was born March 14, 1879."]
        ... )
        >>> print(f"Context Relevance: {result.value}")

    Attributes:
        llm: Modern instructor-based LLM for dual-judge evaluation
        name: The metric name
        allowed_values: Score range (0.0 to 1.0, higher is better)
        max_retries: Maximum retry attempts for invalid ratings
    """

    # Type hints for linter (attributes are set in __init__)
    llm: "InstructorBaseRagasLLM"

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        name: str = "context_relevance",
        max_retries: int = 5,
        **kwargs,
    ):
        """
        Initialize ContextRelevance metric with required components.

        Args:
            llm: Modern instructor-based LLM for dual-judge evaluation
            name: The metric name
            max_retries: Maximum retry attempts for invalid ratings
        """
        # Set attributes explicitly before calling super()
        self.llm = llm
        self.max_retries = max_retries

        # Call super() for validation (without passing llm in kwargs)
        super().__init__(name=name, **kwargs)

    async def ascore(
        self, user_input: str, retrieved_contexts: List[str]
    ) -> MetricResult:
        """
        Calculate context relevance score using dual-judge evaluation.

        Args:
            user_input: The original question
            retrieved_contexts: The retrieved contexts to evaluate for relevance

        Returns:
            MetricResult with context relevance score (0.0-1.0, higher is better)
        """
        # Input validation
        if not user_input:
            raise ValueError(
                "user_input is missing. Please add user_input to the test sample."
            )
        if not retrieved_contexts:
            raise ValueError(
                "retrieved_contexts is missing. Please add retrieved_contexts to the test sample."
            )

        # Handle edge cases like legacy
        context_str = "\n".join(retrieved_contexts)

        if not user_input.strip() or not context_str.strip():
            return MetricResult(value=0.0)

        # Edge case: if user input matches context exactly
        if user_input.strip() == context_str.strip():
            return MetricResult(value=0.0)

        # Edge case: if context is contained in user input
        if context_str.strip() in user_input.strip():
            return MetricResult(value=0.0)

        # Get ratings from both judges with NVIDIA temperature (0.1)
        judge1_rating = await self._get_judge_rating(
            context_relevance_judge1_prompt(user_input, context_str)
        )
        judge2_rating = await self._get_judge_rating(
            context_relevance_judge2_prompt(user_input, context_str)
        )

        # Average the scores (convert from 0,1,2 scale to 0.0-1.0)
        score = self._average_scores(judge1_rating / 2.0, judge2_rating / 2.0)

        return MetricResult(value=float(score))

    async def _get_judge_rating(self, prompt: str) -> float:
        """Get rating from judge with retry logic and NVIDIA temperature."""
        for retry in range(self.max_retries):
            try:
                result = await self.llm.agenerate(prompt, RelevanceRating)
                rating = result.rating

                # Validate rating is in expected range
                if rating in [0, 1, 2]:
                    return float(rating)
                else:
                    if retry < self.max_retries - 1:
                        continue  # Retry if invalid rating
                    else:
                        return float("nan")

            except Exception:
                if retry < self.max_retries - 1:
                    continue  # Retry on exception
                else:
                    return float("nan")

        return float("nan")

    def _average_scores(self, score1: float, score2: float) -> float:
        """Average two judge scores, handling NaN values."""
        if not np.isnan(score1) and not np.isnan(score2):
            return (score1 + score2) / 2.0
        elif not np.isnan(score1):
            return score1
        elif not np.isnan(score2):
            return score2
        else:
            return float("nan")
