"""Base class for all numeric metrics"""

__all__ = ["numeric_metric", "NumericMetric"]

import typing as t
from dataclasses import dataclass

if t.TYPE_CHECKING:
    from ragas.metrics.base import EmbeddingModelType

from .base import SimpleLLMMetric
from .decorator import NumericMetricProtocol, create_metric_decorator
from .validators import NumericValidator


@dataclass(repr=False)
class NumericMetric(SimpleLLMMetric, NumericValidator):
    allowed_values: t.Union[t.Tuple[float, float], range] = (0.0, 1.0)

    def __post_init__(self):
        super().__post_init__()
        # Use the factory to create and mark the model as auto-generated
        from ragas.metrics.base import create_auto_response_model

        self._response_model = create_auto_response_model(
            "NumericResponseModel", reason=(str, ...), value=(float, ...)
        )

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

    @classmethod
    def load(
        cls, path: str, embedding_model: t.Optional["EmbeddingModelType"] = None
    ) -> "NumericMetric":
        """
        Load a NumericMetric from a JSON file.

        Parameters:
        -----------
        path : str
            File path to load from. Supports .gz compressed files.
        embedding_model : Optional[Any]
            Embedding model for DynamicFewShotPrompt. Required if the original used one.

        Returns:
        --------
        NumericMetric
            Loaded metric instance

        Raises:
        -------
        ValueError
            If file cannot be loaded or is not a NumericMetric
        """
        # Validate metric type before loading
        cls._validate_metric_type(path)

        # Load using parent class method
        metric = super().load(path, embedding_model=embedding_model)

        # Additional type check for safety
        if not isinstance(metric, cls):
            raise ValueError(f"Loaded metric is not a {cls.__name__}")

        # Convert allowed_values back to tuple if it's a list (due to JSON serialization)
        if hasattr(metric, "allowed_values") and isinstance(
            metric.allowed_values, list
        ):
            # Ensure it's a 2-element tuple for NumericMetric
            if len(metric.allowed_values) == 2:
                metric.allowed_values = (
                    metric.allowed_values[0],
                    metric.allowed_values[1],
                )
            else:
                metric.allowed_values = tuple(metric.allowed_values)  # type: ignore

        return metric


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

    decorator_factory = create_metric_decorator()
    return decorator_factory(name=name, allowed_values=allowed_values, **metric_params)  # type: ignore[return-value]
