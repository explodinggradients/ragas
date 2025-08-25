"""Base class for ranking metrics"""

__all__ = ["ranking_metric", "RankingMetric"]

import typing as t
from dataclasses import dataclass

from pydantic import Field, create_model

from .decorator import create_metric_decorator
from .llm_based import LLMMetric


@dataclass
class RankingMetric(LLMMetric):
    allowed_values: int = 2

    def __post_init__(self):
        super().__post_init__()
        self._response_model = create_model(
            "RankingResponseModel",
            reason=(str, Field(..., description="Reasoning for the ranking")),
            value=(t.List[str], Field(..., description="List of ranked items")),
        )

    def get_correlation(
        self, gold_labels: t.List[str], predictions: t.List[str]
    ) -> float:
        """
        Calculate the correlation between gold labels and predictions.
        This is a placeholder method and should be implemented based on the specific metric.
        """
        try:
            from sklearn.metrics import cohen_kappa_score
        except ImportError:
            raise ImportError(
                "scikit-learn is required for correlation calculation. "
                "Please install it with `pip install scikit-learn`."
            )

        kappa_scores = []
        for gold_item, prediction in zip(gold_labels, predictions):
            kappa = cohen_kappa_score(gold_item, prediction, weights="quadratic")
            kappa_scores.append(kappa)

        return sum(kappa_scores) / len(kappa_scores) if kappa_scores else 0.0


ranking_metric = create_metric_decorator(RankingMetric)
