"""Rouge Score metric implementation."""

import typing as t

from pydantic import Field

from ragas.metrics.result import MetricResult
from ragas.metrics.v2.base import V2BaseMetric


class RougeScore(V2BaseMetric):
    """
    Calculate ROUGE score between reference and response texts.

    Attributes:
        name: The metric name
        rouge_type: Type of ROUGE metric ("rouge1" for unigrams, "rougeL" for LCS)
        mode: Scoring mode ("fmeasure", "precision", or "recall")
        allowed_values: Score range (0.0 to 1.0)

    Example:
        >>> from ragas.metrics.v2 import RougeScore
        >>>
        >>> metric = RougeScore(rouge_type="rougeL", mode="fmeasure")
        >>> result = await metric.ascore(
        ...     reference="The capital of France is Paris.",
        ...     response="Paris is the capital of France."
        ... )
        >>> print(f"Score: {result.value}")
    """

    name: str = "rouge_score_v2"
    rouge_type: t.Literal["rouge1", "rougeL"] = Field(
        default="rougeL",
        description="Type of ROUGE metric (rouge1 for unigrams, rougeL for LCS)",
    )
    mode: t.Literal["fmeasure", "precision", "recall"] = Field(
        default="fmeasure",
        description="Scoring mode (fmeasure, precision, or recall)",
    )

    async def _ascore_impl(
        self,
        reference: str,
        response: str,
    ) -> MetricResult:
        """
        Calculate ROUGE score asynchronously.

        Args:
            reference: The reference/ground truth text
            response: The response text to evaluate

        Returns:
            MetricResult with ROUGE score (0.0-1.0)

        Raises:
            ImportError: If rouge_score package is not installed
        """
        # Import and check dependencies
        try:
            from rouge_score import rouge_scorer
        except ImportError:
            raise ImportError(
                "rouge_score is required for ROUGE score calculation. "
                "Please install it using `pip install rouge-score`"
            )

        # Calculate ROUGE score
        scorer = rouge_scorer.RougeScorer([self.rouge_type], use_stemmer=True)
        scores = scorer.score(reference, response)
        score_value = getattr(scores[self.rouge_type], self.mode)

        return MetricResult(value=float(score_value))
