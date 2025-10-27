"""SimpleCriteria metric for custom criteria-based evaluation."""

import typing as t
from collections import Counter

from pydantic import BaseModel, Field

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM


class SimpleCriteriaOutput(BaseModel):
    """Output for simple criteria evaluation."""

    reason: str = Field(description="Reason for the scoring")
    score: int = Field(description="The score for the submission")


class SimpleCriteria(BaseMetric):
    """
    Judges submissions using custom criteria with configurable scoring.

    Usage:
        >>> from openai import AsyncOpenAI
        >>> from ragas.llms import llm_factory
        >>> from ragas.metrics.collections import SimpleCriteria
        >>>
        >>> # Setup dependencies
        >>> client = AsyncOpenAI()
        >>> llm = llm_factory("gpt-4o-mini", client=client)
        >>>
        >>> # Create metric instance
        >>> metric = SimpleCriteria(
        ...     name="clarity",
        ...     definition="Is the response clear and easy to understand?",
        ...     llm=llm,
        ... )
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     user_input="What is machine learning?",
        ...     response="Machine learning is a subset of artificial intelligence..."
        ... )
        >>> print(f"Score: {result.value}")

    Attributes:
        llm: Modern instructor-based LLM for evaluation
        name: The metric name
        definition: Criteria to judge the submission
        strictness: Number of times self consistency checks is made (default: 1)
        allowed_values: Score range for numeric validation
    """

    llm: "InstructorBaseRagasLLM"

    def __init__(
        self,
        name: str,
        definition: str,
        llm: "InstructorBaseRagasLLM",
        strictness: int = 1,
        allowed_values: t.Tuple[float, float] = (0.0, 10.0),
        **kwargs,
    ):
        """Initialize SimpleCriteria metric with required components."""
        self.llm = llm
        self.definition = definition
        self.strictness = strictness if strictness % 2 != 0 else strictness + 1

        super().__init__(name=name, allowed_values=allowed_values, **kwargs)

    def _build_prompt(
        self,
        user_input: t.Optional[str] = None,
        response: t.Optional[str] = None,
        retrieved_contexts: t.Optional[t.List[str]] = None,
        reference: t.Optional[str] = None,
        reference_contexts: t.Optional[t.List[str]] = None,
    ) -> str:
        """Build the evaluation prompt from inputs."""
        instruction = f"""Evaluate the input based on the criteria defined.
Criteria Definition: {self.definition}

Provide your evaluation in the following format:
- reason: Brief explanation for your score
- score: Integer score for the submission
"""

        input_parts = []
        if user_input is not None:
            input_parts.append(f"User Input: {user_input}")
        if response is not None:
            input_parts.append(f"Response: {response}")
        if retrieved_contexts is not None and len(retrieved_contexts) > 0:
            contexts_str = "\n".join(f"  - {ctx}" for ctx in retrieved_contexts)
            input_parts.append(f"Retrieved Contexts:\n{contexts_str}")
        if reference is not None:
            input_parts.append(f"Reference: {reference}")
        if reference_contexts is not None and len(reference_contexts) > 0:
            ref_contexts_str = "\n".join(f"  - {ctx}" for ctx in reference_contexts)
            input_parts.append(f"Reference Contexts:\n{ref_contexts_str}")

        input_section = "\n\n".join(input_parts) if input_parts else ""

        return f"{instruction}\n{input_section}"

    async def ascore(
        self,
        user_input: t.Optional[str] = None,
        response: t.Optional[str] = None,
        retrieved_contexts: t.Optional[t.List[str]] = None,
        reference: t.Optional[str] = None,
        reference_contexts: t.Optional[t.List[str]] = None,
    ) -> MetricResult:
        """
        Calculate simple criteria score asynchronously.

        Args:
            user_input: The input to the llm system (optional)
            response: The response from the llm system (optional)
            retrieved_contexts: The retrieved contexts from the llm system (optional)
            reference: The reference answer for evaluation (optional)
            reference_contexts: The reference contexts for evaluation (optional)

        Returns:
            MetricResult with score and reason
        """
        prompt = self._build_prompt(
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts,
            reference=reference,
            reference_contexts=reference_contexts,
        )

        scores = []
        reasons = []

        for _ in range(self.strictness):
            result = await self.llm.agenerate(prompt, SimpleCriteriaOutput)
            scores.append(result.score)
            reasons.append(result.reason)

        if self.strictness > 1:
            score = Counter(scores).most_common(1)[0][0]
            majority_score = score
            reason_idx = scores.index(majority_score)
            reason = reasons[reason_idx]
        else:
            score = scores[0]
            reason = reasons[0]

        return MetricResult(value=float(score), reason=reason)
