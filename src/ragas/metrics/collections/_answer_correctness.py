"""Answer Correctness metric v2 - Modern implementation with function-based prompts."""

import typing as t
from typing import List

import numpy as np
from pydantic import BaseModel

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult
from ragas.prompt.metrics.answer_correctness import (
    correctness_classifier_prompt,
    statement_generator_prompt,
)

if t.TYPE_CHECKING:
    from ragas.embeddings.base import BaseRagasEmbedding
    from ragas.llms.base import InstructorBaseRagasLLM


class StatementGeneratorOutput(BaseModel):
    """Structured output for statement generation."""

    statements: List[str]


class StatementsWithReason(BaseModel):
    """Individual statement with reasoning for classification."""

    statement: str
    reason: str


class ClassificationWithReason(BaseModel):
    """Structured output for TP/FP/FN classification."""

    TP: List[StatementsWithReason]
    FP: List[StatementsWithReason]
    FN: List[StatementsWithReason]


class AnswerCorrectness(BaseMetric):
    """
    Modern v2 implementation of answer correctness evaluation.

    Measures answer correctness as a weighted combination of:
    - Factuality: F1 score from statement-level TP/FP/FN classification
    - Similarity: Semantic similarity between answer and reference

    This implementation uses modern instructor LLMs with structured output and modern embeddings.
    Only supports modern components - legacy wrappers are rejected with clear error messages.

    Usage:
        >>> import instructor
        >>> from openai import AsyncOpenAI
        >>> from ragas.llms.base import instructor_llm_factory
        >>> from ragas.embeddings.base import embedding_factory
        >>> from ragas.metrics.collections import AnswerCorrectness
        >>>
        >>> # Setup dependencies
        >>> client = AsyncOpenAI()
        >>> llm = instructor_llm_factory("openai", client=client, model="gpt-4o-mini")
        >>> embeddings = embedding_factory("openai", model="text-embedding-ada-002", client=client, interface="modern")
        >>>
        >>> # Create metric instance
        >>> metric = AnswerCorrectness(llm=llm, embeddings=embeddings)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     user_input="What is the capital of France?",
        ...     response="Paris is the capital of France and has many museums.",
        ...     reference="Paris is the capital of France."
        ... )
        >>> print(f"Correctness Score: {result.value}")
        >>>
        >>> # Custom weights (more factuality focus)
        >>> factual_metric = AnswerCorrectness(
        ...     llm=llm,
        ...     embeddings=embeddings,
        ...     weights=[0.9, 0.1]
        ... )

    Attributes:
        llm: Modern instructor-based LLM for statement generation and classification
        embeddings: Modern embeddings model for similarity calculation
        name: The metric name
        weights: [factuality_weight, similarity_weight] - must sum to > 0
        beta: F-beta score parameter (β>1 favors recall, β<1 favors precision)
        allowed_values: Score range (0.0 to 1.0)
    """

    # Type hints for linter (attributes are set in __init__)
    llm: "InstructorBaseRagasLLM"
    embeddings: "BaseRagasEmbedding"

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        embeddings: "BaseRagasEmbedding",
        name: str = "answer_correctness",
        weights: List[float] = [0.75, 0.25],
        beta: float = 1.0,
        **kwargs,
    ):
        """
        Initialize AnswerCorrectness metric with required components.

        Args:
            llm: Modern instructor-based LLM for statement generation and classification
            embeddings: Modern embeddings model for similarity calculation
            weights: [factuality_weight, similarity_weight]. Must sum to > 0.
            beta: F-beta score parameter. β>1 favors recall, β<1 favors precision.
        """
        # Set attributes explicitly before calling super()
        self.llm = llm
        self.embeddings = embeddings
        self.weights = weights
        self.beta = beta

        # Validate weights
        if len(weights) != 2:
            raise ValueError(
                "Expects a list of two weights. First for factuality, second for semantic similarity"
            )
        if all([w == 0 for w in weights]):
            raise ValueError("At least one weight must be non-zero")
        if not all([w >= 0 for w in weights]):
            raise ValueError("Weights must be non-negative")

        # Validate beta
        if not isinstance(beta, float):
            raise ValueError(
                "Beta must be a float. A beta > 1 gives more weight to recall, while beta < 1 favors precision."
            )

        # Call super() for validation (without passing llm/embeddings in kwargs)
        super().__init__(name=name, **kwargs)

    async def ascore(
        self, user_input: str, response: str, reference: str
    ) -> MetricResult:
        """
        Calculate answer correctness score.

        Components are guaranteed to be validated and non-None by the base class.

        Args:
            user_input: The original question
            response: The answer to evaluate
            reference: The ground truth reference

        Returns:
            MetricResult with correctness score (0.0-1.0)
        """
        # Step 1: Generate statements from both response and reference
        response_statements = await self._generate_statements(user_input, response)
        reference_statements = await self._generate_statements(user_input, reference)

        # Step 2: Calculate factuality score via TP/FP/FN classification
        if response_statements and reference_statements:
            classification = await self._classify_statements(
                user_input, response_statements, reference_statements
            )
            factuality_score = self._compute_f1_score(classification)
        else:
            # If no statements generated, assume perfect match
            factuality_score = 1.0

        # Step 3: Calculate semantic similarity score
        if self.weights[1] == 0:
            similarity_score = 0.0
        else:
            similarity_score = await self._calculate_similarity(response, reference)

        # Step 4: Combine scores with weighted average
        final_score = np.average(
            [factuality_score, similarity_score],
            weights=self.weights,
        )

        return MetricResult(value=float(final_score))

    async def _generate_statements(self, question: str, text: str) -> List[str]:
        """Generate atomic statements from text using the statement generator prompt."""
        prompt = statement_generator_prompt(question, text)
        # Use deterministic defaults set in LLM constructor
        result = await self.llm.agenerate(prompt, StatementGeneratorOutput)
        return result.statements

    async def _classify_statements(
        self,
        question: str,
        answer_statements: List[str],
        ground_truth_statements: List[str],
    ) -> ClassificationWithReason:
        """Classify statements as TP/FP/FN using the correctness classifier prompt with strict behavior."""
        prompt = correctness_classifier_prompt(
            question, answer_statements, ground_truth_statements
        )
        # Use deterministic defaults set in LLM constructor
        classification = await self.llm.agenerate(prompt, ClassificationWithReason)
        return classification

    def _compute_f1_score(self, classification: ClassificationWithReason) -> float:
        """Compute F1 score from TP/FP/FN classification."""
        tp = len(classification.TP)
        fp = len(classification.FP)
        fn = len(classification.FN)

        # Calculate precision and recall
        if tp + fp == 0:
            precision = 1.0 if fn == 0 else 0.0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 1.0 if fp == 0 else 0.0
        else:
            recall = tp / (tp + fn)

        # Calculate F-beta score
        if precision + recall == 0:
            return 0.0

        beta_squared = self.beta**2
        f_score = (
            (1 + beta_squared)
            * (precision * recall)
            / (beta_squared * precision + recall)
        )

        return float(f_score)

    async def _calculate_similarity(self, response: str, reference: str) -> float:
        """Calculate semantic similarity between response and reference using embeddings."""
        # Get embeddings for both texts
        response_embedding = np.asarray(
            await self.embeddings.aembed_text(response)
        ).reshape(1, -1)
        reference_embedding = np.asarray(
            await self.embeddings.aembed_text(reference)
        ).reshape(1, -1)

        # Calculate cosine similarity
        norm_response = np.linalg.norm(response_embedding, axis=1)
        norm_reference = np.linalg.norm(reference_embedding, axis=1)

        if norm_response == 0 or norm_reference == 0:
            return 0.0

        cosine_similarity = np.dot(response_embedding, reference_embedding.T)[0, 0] / (
            norm_response[0] * norm_reference[0]
        )

        return float(cosine_similarity)
