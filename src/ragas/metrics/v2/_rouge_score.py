"""ROUGE Score metric implementation using the v2 decorator pattern."""

import typing as t

from ragas.metrics.numeric import numeric_metric
from ragas.metrics.result import MetricResult


@numeric_metric(name="rouge_score", allowed_values=(0.0, 1.0))
async def rouge_score(
    reference: str,
    response: str,
    rouge_type: t.Literal["rouge1", "rougeL"] = "rougeL",
    mode: t.Literal["fmeasure", "precision", "recall"] = "fmeasure",
) -> MetricResult:
    """
    Calculate ROUGE score between reference and response texts.

    Args:
        reference: The reference/ground truth text
        response: The response text to evaluate
        rouge_type: Type of ROUGE metric ("rouge1" for unigrams, "rougeL" for LCS)
        mode: Scoring mode ("fmeasure", "precision", or "recall")

    Returns:
        MetricResult with score value (0.0-1.0)
    """
    # Import and check dependencies
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise ImportError(
            "rouge_score is required for rouge score. Please install it using `pip install rouge_score`"
        )

    # Calculate ROUGE score
    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    scores = scorer.score(reference, response)
    score_value = getattr(scores[rouge_type], mode)

    return MetricResult(value=float(score_value))
