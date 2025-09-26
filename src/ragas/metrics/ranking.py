"""Base class for ranking metrics"""

__all__ = ["ranking_metric", "RankingMetric"]

import typing as t
from dataclasses import dataclass

from pydantic import Field, create_model

from .base import SimpleLLMMetric
from .decorator import RankingMetricProtocol, create_metric_decorator
from .validators import RankingValidator


@dataclass
class RankingMetric(SimpleLLMMetric, RankingValidator):
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

    @classmethod
    def load(
        cls, path: str, embedding_model: t.Optional[t.Any] = None
    ) -> "RankingMetric":
        """
        Load a RankingMetric from a JSON file.

        Parameters:
        -----------
        path : str
            File path to load from. Supports .gz compressed files.
        embedding_model : Optional[Any]
            Embedding model for DynamicFewShotPrompt. Required if the original used one.

        Returns:
        --------
        RankingMetric
            Loaded metric instance

        Raises:
        -------
        ValueError
            If file cannot be loaded or is not a RankingMetric
        """
        # Load using parent class method
        metric = super().load(path, embedding_model=embedding_model)

        # Validate it's the correct type
        if not isinstance(metric, cls):
            raise ValueError(f"Loaded metric is not a {cls.__name__}")

        return metric


def ranking_metric(
    *,
    name: t.Optional[str] = None,
    allowed_values: t.Optional[int] = None,
    **metric_params,
) -> t.Callable[[t.Callable[..., t.Any]], RankingMetricProtocol]:
    """
    Decorator for creating ranking metrics.

    Args:
        name: Optional name for the metric (defaults to function name)
        allowed_values: Expected length of the returned ranking list
        **metric_params: Additional parameters for the metric

    Returns:
        A decorator that transforms a function into a RankingMetric instance
    """
    if allowed_values is None:
        allowed_values = 2

    decorator_factory = create_metric_decorator()
    return decorator_factory(name=name, allowed_values=allowed_values, **metric_params)
