"""Base class for all numeric metrics"""

__all__ = ["numeric_metric", "NumericMetric"]

import typing as t
from dataclasses import dataclass

from pydantic import create_model

from . import Metric
from .decorator import create_metric_decorator


@dataclass
class NumericMetric(Metric):
    allowed_values: t.Union[t.Tuple[float, float], range] = (0.0, 1.0)

    def __post_init__(self):
        super().__post_init__()
        self._response_model = create_model("response_model", value=(float, ...))

    def get_correlation(
        self, gold_labels: t.List[float], predictions: t.List[float]
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
        return pearsonr(gold_labels, predictions)[0]


numeric_metric = create_metric_decorator(NumericMetric)
