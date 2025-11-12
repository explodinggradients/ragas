"""Context Recall metric v2 - Class-based implementation with modern components."""

import logging
import typing as t
from typing import List

import numpy as np
from pydantic import BaseModel

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._string import DistanceMeasure, NonLLMStringSimilarity
from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult
from ragas.prompt.metrics.context_recall import context_recall_prompt

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM

logger = logging.getLogger(__name__)


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
        """
        Initialize ContextRecall metric with required components.

        Args:
            llm: Modern instructor-based LLM for classification
            name: The metric name (default: "context_recall")
            **kwargs: Additional arguments passed to BaseMetric
        """
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
        # Input validation
        if not user_input:
            raise ValueError("user_input cannot be empty")
        if not reference:
            raise ValueError("reference cannot be empty")
        if not retrieved_contexts:
            raise ValueError("retrieved_contexts cannot be empty")

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


class NonLLMContextRecall(BaseMetric):
    """
    Evaluate context recall using string similarity without LLM.

    Compares retrieved contexts with reference contexts using string similarity metrics.
    A reference context is considered recalled if it has sufficient similarity with
    at least one retrieved context.

    This implementation provides deterministic evaluation without requiring LLM components.

    Usage:
        >>> from ragas.metrics.collections import NonLLMContextRecall
        >>>
        >>> metric = NonLLMContextRecall(threshold=0.5)
        >>>
        >>> result = await metric.ascore(
        ...     retrieved_contexts=["Albert Einstein was a physicist"],
        ...     reference_contexts=["Einstein was a theoretical physicist"]
        ... )
        >>> print(f"Context Recall: {result.value}")

    Attributes:
        name: The metric name
        threshold: Similarity threshold for considering a context as recalled (default: 0.5)
        distance_measure: The string distance measure to use (default: LEVENSHTEIN)
        allowed_values: Score range (0.0 to 1.0)
    """

    def __init__(
        self,
        name: str = "non_llm_context_recall",
        threshold: float = 0.5,
        distance_measure: DistanceMeasure = DistanceMeasure.LEVENSHTEIN,
        **kwargs,
    ):
        """
        Initialize NonLLMContextRecall metric.

        Args:
            name: The metric name
            threshold: Similarity threshold (0.0-1.0) for considering a context recalled
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
        Calculate context recall score using string similarity.

        Args:
            retrieved_contexts: List of retrieved context strings
            reference_contexts: List of reference context strings

        Returns:
            MetricResult with recall score (0.0-1.0, higher is better)
        """
        if not retrieved_contexts:
            raise ValueError("retrieved_contexts cannot be empty")
        if not reference_contexts:
            raise ValueError("reference_contexts cannot be empty")

        scores = []
        for ref in reference_contexts:
            max_similarity = 0.0
            for rc in retrieved_contexts:
                # Use the distance measure to compute similarity
                similarity = await self._distance_measure.single_turn_ascore(
                    SingleTurnSample(reference=rc, response=ref),
                    callbacks=None,
                )
                max_similarity = max(max_similarity, similarity)
            scores.append(max_similarity)

        # Compute recall: proportion of reference contexts above threshold
        recalled = [1 if score > self.threshold else 0 for score in scores]
        score = sum(recalled) / len(recalled) if recalled else np.nan

        return MetricResult(value=float(score))


class IDBasedContextRecall(BaseMetric):
    """
    Evaluate context recall by comparing retrieved and reference context IDs.

    Directly compares retrieved context IDs with reference context IDs.
    The score represents the proportion of reference IDs that were successfully retrieved.

    This implementation works with both string and integer IDs and provides
    deterministic evaluation without requiring LLM components.

    Usage:
        >>> from ragas.metrics.collections import IDBasedContextRecall
        >>>
        >>> metric = IDBasedContextRecall()
        >>>
        >>> result = await metric.ascore(
        ...     retrieved_context_ids=["doc1", "doc2", "doc3"],
        ...     reference_context_ids=["doc1", "doc2", "doc4"]
        ... )
        >>> print(f"Context Recall: {result.value}")  # 0.667

    Attributes:
        name: The metric name
        allowed_values: Score range (0.0 to 1.0)
    """

    def __init__(
        self,
        name: str = "id_based_context_recall",
        **kwargs,
    ):
        """
        Initialize IDBasedContextRecall metric.

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
        Calculate context recall score based on ID matching.

        Args:
            retrieved_context_ids: List of retrieved context IDs (strings or integers)
            reference_context_ids: List of reference context IDs (strings or integers)

        Returns:
            MetricResult with recall score (0.0-1.0, higher is better)
        """
        if not retrieved_context_ids:
            raise ValueError("retrieved_context_ids cannot be empty")
        if not reference_context_ids:
            raise ValueError("reference_context_ids cannot be empty")

        # Convert all IDs to strings for consistent comparison
        retrieved_ids_set = set(str(id_) for id_ in retrieved_context_ids)
        reference_ids_set = set(str(id_) for id_ in reference_context_ids)

        # Calculate how many reference IDs appear in retrieved IDs
        hits = sum(
            1 for ref_id in reference_ids_set if str(ref_id) in retrieved_ids_set
        )

        # Calculate recall score
        total_refs = len(reference_ids_set)
        score = hits / total_refs if total_refs > 0 else np.nan

        if np.isnan(score):
            logger.warning(
                "No reference context IDs provided, cannot calculate recall."
            )

        return MetricResult(value=float(score))
