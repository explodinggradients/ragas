"""Base class for all numeric metrics"""

__all__ = ["numeric_metric", "NumericMetric"]

import typing as t
from dataclasses import dataclass

from pydantic import create_model

from .decorator import NumericMetricProtocol, create_metric_decorator
from .llm_based import LLMMetric


@dataclass
class NumericMetric(LLMMetric):
    allowed_values: t.Union[t.Tuple[float, float], range] = (0.0, 1.0)

    def __post_init__(self):
        super().__post_init__()
        self._response_model = create_model("response_model", value=(float, ...))

    def get_correlation(
        self, gold_labels: t.List[str], predictions: t.List[str]
    ) -> float:
        """
        Calculate the correlation between gold labels and predictions.
        This is a placeholder method and should be implemented based on the specific metric.
        """
        try:
            from scipy.stats import pearsonr
        except ImportError:
            raise ImportError(
                "scipy is required for correlation calculation. "
                "Please install it with `pip install scipy`."
            )
        # Convert strings to floats for correlation calculation
        gold_floats = [float(x) for x in gold_labels]
        pred_floats = [float(x) for x in predictions]
        result = pearsonr(gold_floats, pred_floats)
        # pearsonr returns (correlation, p-value) tuple
        correlation = t.cast(float, result[0])
        return correlation


def numeric_metric(
    *,
    name: t.Optional[str] = None,
    allowed_values: t.Optional[t.Union[t.Tuple[float, float], range]] = None,
    **metric_params,
) -> t.Callable[[t.Callable[..., t.Any]], NumericMetricProtocol]:
    """
    Decorator for creating numeric metrics.

    Args:
        name: Optional name for the metric (defaults to function name)
        allowed_values: Tuple specifying (min, max) range or range object for valid values
        **metric_params: Additional parameters for the metric

    Returns:
        A decorator that transforms a function into a NumericMetric instance
    """
    if allowed_values is None:
        allowed_values = (0.0, 1.0)

    decorator_factory = create_metric_decorator(NumericMetric)
    return decorator_factory(name=name, allowed_values=allowed_values, **metric_params)
