"""Base class from which all discrete metrics should inherit."""

__all__ = ["discrete_metric", "DiscreteMetric"]

import typing as t
from dataclasses import dataclass, field

from pydantic import create_model

from .decorator import create_metric_decorator
from .llm_based import LLMMetric


@dataclass
class DiscreteMetric(LLMMetric):
    allowed_values: t.List[str] = field(default_factory=lambda: ["pass", "fail"])

    def __post_init__(self):
        super().__post_init__()
        values = tuple(self.allowed_values)
        self._response_model = create_model(
            "response_model", value=(t.Literal[values], ...), reason=(str, ...)
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
        return cohen_kappa_score(gold_labels, predictions)


discrete_metric = create_metric_decorator(DiscreteMetric)
