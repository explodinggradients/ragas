"""Context Recall metric v2 - Class-based implementation with modern components."""

import typing as t

import numpy as np
from pydantic import BaseModel

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult
from ragas.prompt.metrics.context_recall import context_recall_prompt

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM


class ContextRecallClassification(BaseModel):
    """Structured output for a single statement classification."""

    statement: str
    reason: str
    attributed: int


class ContextRecallOutput(BaseModel):
    """Structured output for context recall classifications."""

    classifications: t.List[ContextRecallClassification]


class ContextRecall(BaseMetric):
    """
    Evaluate context recall by classifying if statements can be attributed to context.

    This implementation uses modern instructor LLMs with structured output.
    Only supports modern components - legacy wrappers are rejected with clear error messages.

    Usage:
        >>> import instructor
        >>> from openai import AsyncOpenAI
        >>> from ragas.llms.base import instructor_llm_factory
        >>> from ragas.metrics.collections import ContextRecall
        >>>
        >>> # Setup dependencies
        >>> client = AsyncOpenAI()
        >>> llm = instructor_llm_factory("openai", client=client, model="gpt-4o-mini")
        >>>
        >>> # Create metric instance
        >>> metric = ContextRecall(llm=llm)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     user_input="What is the capital of France?",
        ...     retrieved_contexts=["Paris is the capital of France."],
        ...     reference="Paris is the capital and largest city of France."
        ... )
        >>> print(f"Score: {result.value}")
        >>>
        >>> # Batch evaluation
        >>> results = await metric.abatch_score([
        ...     {"user_input": "Q1", "retrieved_contexts": ["C1"], "reference": "A1"},
        ...     {"user_input": "Q2", "retrieved_contexts": ["C2"], "reference": "A2"},
        ... ])

    Attributes:
        llm: Modern instructor-based LLM for classification
        name: The metric name
        allowed_values: Score range (0.0 to 1.0)
    """

    # Type hints for linter (attributes are set in __init__)
    llm: "InstructorBaseRagasLLM"

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        name: str = "context_recall",
        **kwargs,
    ):
        """Initialize ContextRecall metric with required components."""
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
        Calculate context recall score asynchronously.

        Components are guaranteed to be validated and non-None by the base class.

        Args:
            user_input: The original question
            retrieved_contexts: List of retrieved context strings
            reference: The reference answer to evaluate

        Returns:
            MetricResult with recall score (0.0-1.0)
        """
        # Combine contexts into a single string
        context = "\n".join(retrieved_contexts) if retrieved_contexts else ""

        # Generate prompt
        prompt = context_recall_prompt(
            question=user_input, context=context, answer=reference
        )

        # Get classifications from LLM
        result = await self.llm.agenerate(prompt, ContextRecallOutput)

        # Calculate score
        if not result.classifications:
            return MetricResult(value=np.nan)

        # Count attributions
        attributions = [c.attributed for c in result.classifications]
        score = sum(attributions) / len(attributions) if attributions else np.nan

        return MetricResult(value=float(score))
