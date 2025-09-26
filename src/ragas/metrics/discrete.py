"""Base class from which all discrete metrics should inherit."""

__all__ = ["discrete_metric", "DiscreteMetric"]

import typing as t
from dataclasses import dataclass, field

from pydantic import create_model

from .base import SimpleLLMMetric
from .decorator import DiscreteMetricProtocol, create_metric_decorator
from .validators import DiscreteValidator


@dataclass
class DiscreteMetric(SimpleLLMMetric, DiscreteValidator):
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

    @classmethod
    def load(
        cls, path: str, embedding_model: t.Optional[t.Any] = None
    ) -> "DiscreteMetric":
        """
        Load a DiscreteMetric from a JSON file.

        Parameters:
        -----------
        path : str
            File path to load from. Supports .gz compressed files.
        embedding_model : Optional[Any]
            Embedding model for DynamicFewShotPrompt. Required if the original used one.

        Returns:
        --------
        DiscreteMetric
            Loaded metric instance

        Raises:
        -------
        ValueError
            If file cannot be loaded or is not a DiscreteMetric
        """
        # Load using parent class method
        metric = super().load(path, embedding_model=embedding_model)

        # Validate it's the correct type
        if not isinstance(metric, cls):
            raise ValueError(f"Loaded metric is not a {cls.__name__}")

        return metric


def discrete_metric(
    *,
    name: t.Optional[str] = None,
    allowed_values: t.Optional[t.List[str]] = None,
    **metric_params,
) -> t.Callable[[t.Callable[..., t.Any]], DiscreteMetricProtocol]:
    """
    Decorator for creating discrete metrics.

    Args:
        name: Optional name for the metric (defaults to function name)
        allowed_values: List of allowed string values for the metric
        **metric_params: Additional parameters for the metric

    Returns:
        A decorator that transforms a function into a DiscreteMetric instance
    """
    if allowed_values is None:
        allowed_values = ["pass", "fail"]

    decorator_factory = create_metric_decorator()
    return decorator_factory(name=name, allowed_values=allowed_values, **metric_params)
