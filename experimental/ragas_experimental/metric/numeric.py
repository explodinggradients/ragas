"""Base class for all numeric metrics"""

__all__ = ["numeric_metric", "NumericMetric"]

import typing as t
from dataclasses import dataclass

from pydantic import create_model

from . import Metric
from .decorator import create_metric_decorator


@dataclass
class NumericMetric(Metric):
    range: t.Tuple[float, float] = (0.0, 1.0)

    def __post_init__(self):
        super().__post_init__()
        self._response_model = create_model("response_model", result=(float, ...))


numeric_metric = create_metric_decorator(NumericMetric)
