"""Context Precision metrics v2 - Modern implementation with function-based prompts."""

import logging
import typing as t
from typing import List

import numpy as np
from pydantic import BaseModel

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._string import DistanceMeasure, NonLLMStringSimilarity
from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult
from ragas.prompt.metrics.context_precision import (
    context_precision_with_reference_prompt,
    context_precision_without_reference_prompt,
)

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM

logger = logging.getLogger(__name__)


class ContextPrecisionOutput(BaseModel):
    """Structured output for context precision evaluation."""

    reason: str
    verdict: int


class ContextPrecisionWithReference(BaseMetric):
    """
    Modern v2 implementation of context precision with reference.

    Evaluates whether retrieved contexts are useful for answering a question by comparing
    each context against a reference answer. The metric calculates average precision
    based on the usefulness verdicts from an LLM.

    This implementation uses modern instructor LLMs with structured output.
    Only supports modern components - legacy wrappers are rejected with clear error messages.

    Usage:
        >>> import openai
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import ContextPrecisionWithReference
        >>>
        >>> # Setup dependencies
        >>> client = openai.AsyncOpenAI()
        >>> llm = llm_factory("gpt-4o-mini", client=client)
        >>>
        >>> # Create metric instance
        >>> metric = ContextPrecisionWithReference(llm=llm)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     user_input="What is the capital of France?",
        ...     reference="Paris is the capital of France.",
        ...     retrieved_contexts=["Paris is the capital and largest city of France.", "Berlin is the capital of Germany."]
        ... )
        >>> print(f"Context Precision: {result.value}")

    Attributes:
        llm: Modern instructor-based LLM for context evaluation
        name: The metric name
        allowed_values: Score range (0.0 to 1.0, higher is better)
    """

    # Type hints for linter (attributes are set in __init__)
    llm: "InstructorBaseRagasLLM"

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        name: str = "context_precision_with_reference",
        **kwargs,
    ):
        """
        Initialize ContextPrecisionWithReference metric with required components.

        Args:
            llm: Modern instructor-based LLM for context evaluation
            name: The metric name
        """
        # Set attributes explicitly before calling super()
        self.llm = llm

        # Call super() for validation (without passing llm in kwargs)
        super().__init__(name=name, **kwargs)

    async def ascore(
        self, user_input: str, reference: str, retrieved_contexts: List[str]
    ) -> MetricResult:
        """
        Calculate context precision score using reference.

        Args:
            user_input: The question being asked
            reference: The reference answer to compare against
            retrieved_contexts: The retrieved contexts to evaluate

        Returns:
            MetricResult with context precision score (0.0-1.0, higher is better)
        """
        # Input validation
        if not user_input:
            raise ValueError("user_input cannot be empty")
        if not reference:
            raise ValueError("reference cannot be empty")
        if not retrieved_contexts:
            raise ValueError("retrieved_contexts cannot be empty")

        # Evaluate each retrieved context
        verdicts = []
        for context in retrieved_contexts:
            prompt = context_precision_with_reference_prompt(
                user_input, context, reference
            )
            result = await self.llm.agenerate(prompt, ContextPrecisionOutput)
            verdicts.append(result.verdict)

        # Calculate average precision
        score = self._calculate_average_precision(verdicts)
        return MetricResult(value=float(score))

    def _calculate_average_precision(self, verdicts: List[int]) -> float:
        """Calculate average precision from binary verdicts. Matches legacy logic exactly."""
        verdict_list = verdicts
        denominator = sum(verdict_list) + 1e-10
        numerator = sum(
            [
                (sum(verdict_list[: i + 1]) / (i + 1)) * verdict_list[i]
                for i in range(len(verdict_list))
            ]
        )
        score = numerator / denominator

        if np.isnan(score):
            # Match legacy warning behavior
            import logging

            logging.warning(
                "Invalid response format. Expected a list of dictionaries with keys 'verdict'"
            )

        return score


class ContextPrecisionWithoutReference(BaseMetric):
    """
    Modern v2 implementation of context precision without reference.

    Evaluates whether retrieved contexts are useful for answering a question by comparing
    each context against the generated response. The metric calculates average precision
    based on the usefulness verdicts from an LLM.

    This implementation uses modern instructor LLMs with structured output.
    Only supports modern components - legacy wrappers are rejected with clear error messages.

    Usage:
        >>> import openai
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import ContextPrecisionWithoutReference
        >>>
        >>> # Setup dependencies
        >>> client = openai.AsyncOpenAI()
        >>> llm = llm_factory("gpt-4o-mini", client=client)
        >>>
        >>> # Create metric instance
        >>> metric = ContextPrecisionWithoutReference(llm=llm)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     user_input="What is the capital of France?",
        ...     response="Paris is the capital of France.",
        ...     retrieved_contexts=["Paris is the capital and largest city of France.", "Berlin is the capital of Germany."]
        ... )
        >>> print(f"Context Precision: {result.value}")

    Attributes:
        llm: Modern instructor-based LLM for context evaluation
        name: The metric name
        allowed_values: Score range (0.0 to 1.0, higher is better)
    """

    # Type hints for linter (attributes are set in __init__)
    llm: "InstructorBaseRagasLLM"

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        name: str = "context_precision_without_reference",
        **kwargs,
    ):
        """
        Initialize ContextPrecisionWithoutReference metric with required components.

        Args:
            llm: Modern instructor-based LLM for context evaluation
            name: The metric name
        """
        # Set attributes explicitly before calling super()
        self.llm = llm

        # Call super() for validation (without passing llm in kwargs)
        super().__init__(name=name, **kwargs)

    async def ascore(
        self, user_input: str, response: str, retrieved_contexts: List[str]
    ) -> MetricResult:
        """
        Calculate context precision score using response.

        Args:
            user_input: The question being asked
            response: The response that was generated
            retrieved_contexts: The retrieved contexts to evaluate

        Returns:
            MetricResult with context precision score (0.0-1.0, higher is better)
        """
        # Input validation
        if not user_input:
            raise ValueError("user_input cannot be empty")
        if not response:
            raise ValueError("response cannot be empty")
        if not retrieved_contexts:
            raise ValueError("retrieved_contexts cannot be empty")

        # Evaluate each retrieved context
        verdicts = []
        for context in retrieved_contexts:
            prompt = context_precision_without_reference_prompt(
                user_input, context, response
            )
            result = await self.llm.agenerate(prompt, ContextPrecisionOutput)
            verdicts.append(result.verdict)

        # Calculate average precision
        score = self._calculate_average_precision(verdicts)
        return MetricResult(value=float(score))

    def _calculate_average_precision(self, verdicts: List[int]) -> float:
        """Calculate average precision from binary verdicts. Matches legacy logic exactly."""
        verdict_list = verdicts
        denominator = sum(verdict_list) + 1e-10
        numerator = sum(
            [
                (sum(verdict_list[: i + 1]) / (i + 1)) * verdict_list[i]
                for i in range(len(verdict_list))
            ]
        )
        score = numerator / denominator

        if np.isnan(score):
            # Match legacy warning behavior
            import logging

            logging.warning(
                "Invalid response format. Expected a list of dictionaries with keys 'verdict'"
            )

        return score


class ContextPrecision(ContextPrecisionWithReference):
    """
    Modern v2 wrapper for ContextPrecisionWithReference with shorter name.

    This is a simple wrapper that provides the legacy "context_precision" name
    while using the modern V2 implementation underneath.

    Usage:
        >>> import openai
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import ContextPrecision
        >>>
        >>> # Setup dependencies
        >>> client = openai.AsyncOpenAI()
        >>> llm = llm_factory("gpt-4o-mini", client=client)
        >>>
        >>> # Create metric instance (same as ContextPrecisionWithReference)
        >>> metric = ContextPrecision(llm=llm)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     user_input="What is the capital of France?",
        ...     reference="Paris is the capital of France.",
        ...     retrieved_contexts=["Paris is the capital and largest city of France."]
        ... )
    """

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        **kwargs,
    ):
        """Initialize ContextPrecision with the legacy default name."""
        super().__init__(llm, name="context_precision", **kwargs)


class ContextUtilization(ContextPrecisionWithoutReference):
    """
    Modern v2 wrapper for ContextPrecisionWithoutReference with shorter name.

    This is a simple wrapper that provides the legacy "context_utilization" name
    while using the modern V2 implementation underneath.

    Usage:
        >>> import openai
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import ContextUtilization
        >>>
        >>> # Setup dependencies
        >>> client = openai.AsyncOpenAI()
        >>> llm = llm_factory("gpt-4o-mini", client=client)
        >>>
        >>> # Create metric instance (same as ContextPrecisionWithoutReference)
        >>> metric = ContextUtilization(llm=llm)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     user_input="What is the capital of France?",
        ...     response="Paris is the capital of France.",
        ...     retrieved_contexts=["Paris is the capital and largest city of France."]
        ... )
    """

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        **kwargs,
    ):
        """Initialize ContextUtilization with the legacy default name."""
        super().__init__(llm, name="context_utilization", **kwargs)


class NonLLMContextPrecisionWithReference(BaseMetric):
    """
    Evaluate context precision using string similarity without LLM.

    Compares retrieved contexts with reference contexts using string similarity.
    A retrieved context is considered useful if it has sufficient similarity with
    at least one reference context. Calculates average precision based on usefulness.

    This implementation provides deterministic evaluation without requiring LLM components.

    Usage:
        >>> from ragas.metrics.collections import NonLLMContextPrecisionWithReference
        >>>
        >>> metric = NonLLMContextPrecisionWithReference(threshold=0.5)
        >>>
        >>> result = await metric.ascore(
        ...     retrieved_contexts=["Albert Einstein was a physicist"],
        ...     reference_contexts=["Einstein was a theoretical physicist"]
        ... )
        >>> print(f"Context Precision: {result.value}")

    Attributes:
        name: The metric name
        threshold: Similarity threshold for considering a context as useful (default: 0.5)
        distance_measure: The string distance measure to use (default: LEVENSHTEIN)
        allowed_values: Score range (0.0 to 1.0)
    """

    def __init__(
        self,
        name: str = "non_llm_context_precision_with_reference",
        threshold: float = 0.5,
        distance_measure: DistanceMeasure = DistanceMeasure.LEVENSHTEIN,
        **kwargs,
    ):
        """
        Initialize NonLLMContextPrecisionWithReference metric.

        Args:
            name: The metric name
            threshold: Similarity threshold (0.0-1.0) for considering a context useful
            distance_measure: The string distance measure to use
            **kwargs: Additional arguments passed to BaseMetric
        """
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self._distance_measure = NonLLMStringSimilarity(
            distance_measure=distance_measure
        )

    async def ascore(
        self,
        retrieved_contexts: List[str],
        reference_contexts: List[str],
    ) -> MetricResult:
        """
        Calculate context precision score using string similarity.

        Args:
            retrieved_contexts: List of retrieved context strings
            reference_contexts: List of reference context strings

        Returns:
            MetricResult with precision score (0.0-1.0, higher is better)
        """
        if not retrieved_contexts:
            raise ValueError("retrieved_contexts cannot be empty")
        if not reference_contexts:
            raise ValueError("reference_contexts cannot be empty")

        scores = []
        for rc in retrieved_contexts:
            max_similarity = 0.0
            for ref in reference_contexts:
                # Use the distance measure to compute similarity
                similarity = await self._distance_measure.single_turn_ascore(
                    SingleTurnSample(reference=rc, response=ref),
                    callbacks=None,
                )
                max_similarity = max(max_similarity, similarity)
            scores.append(max_similarity)

        # Convert to binary verdicts based on threshold
        verdicts = [1 if score >= self.threshold else 0 for score in scores]

        # Calculate average precision
        score = self._calculate_average_precision(verdicts)
        return MetricResult(value=float(score))

    def _calculate_average_precision(self, verdict_list: List[int]) -> float:
        """Calculate average precision from binary verdicts."""
        if not verdict_list:
            return np.nan

        denominator = sum(verdict_list) + 1e-10
        numerator = sum(
            [
                (sum(verdict_list[: i + 1]) / (i + 1)) * verdict_list[i]
                for i in range(len(verdict_list))
            ]
        )
        score = numerator / denominator
        return score


class IDBasedContextPrecision(BaseMetric):
    """
    Evaluate context precision by comparing retrieved and reference context IDs.

    Directly compares retrieved context IDs with reference context IDs.
    The score represents the proportion of retrieved IDs that are actually relevant
    (present in reference IDs). Calculates average precision based on relevance.

    This implementation works with both string and integer IDs and provides
    deterministic evaluation without requiring LLM components.

    Usage:
        >>> from ragas.metrics.collections import IDBasedContextPrecision
        >>>
        >>> metric = IDBasedContextPrecision()
        >>>
        >>> result = await metric.ascore(
        ...     retrieved_context_ids=["doc1", "doc2", "doc3"],
        ...     reference_context_ids=["doc1", "doc2", "doc4"]
        ... )
        >>> print(f"Context Precision: {result.value}")  # 0.667

    Attributes:
        name: The metric name
        allowed_values: Score range (0.0 to 1.0)
    """

    def __init__(
        self,
        name: str = "id_based_context_precision",
        **kwargs,
    ):
        """
        Initialize IDBasedContextPrecision metric.

        Args:
            name: The metric name
            **kwargs: Additional arguments passed to BaseMetric
        """
        super().__init__(name=name, **kwargs)

    async def ascore(
        self,
        retrieved_context_ids: t.Union[t.List[str], t.List[int]],
        reference_context_ids: t.Union[t.List[str], t.List[int]],
    ) -> MetricResult:
        """
        Calculate context precision score based on ID matching.

        Args:
            retrieved_context_ids: List of retrieved context IDs (strings or integers)
            reference_context_ids: List of reference context IDs (strings or integers)

        Returns:
            MetricResult with precision score (0.0-1.0, higher is better)
        """
        if not retrieved_context_ids:
            raise ValueError("retrieved_context_ids cannot be empty")
        if not reference_context_ids:
            raise ValueError("reference_context_ids cannot be empty")

        # Convert all IDs to strings for consistent comparison
        retrieved_ids_set = set(str(id_) for id_ in retrieved_context_ids)
        reference_ids_set = set(str(id_) for id_ in reference_context_ids)

        # Count how many retrieved IDs match reference IDs
        hits = sum(
            1 for ret_id in retrieved_ids_set if str(ret_id) in reference_ids_set
        )

        # For precision: relevant retrieved / total retrieved
        total_retrieved = len(retrieved_ids_set)
        if total_retrieved == 0:
            logger.warning(
                "No retrieved context IDs provided, cannot calculate precision."
            )
            return MetricResult(value=np.nan)

        score = hits / total_retrieved
        return MetricResult(value=float(score))
