"""Base class for ranking metrics"""

__all__ = ["ranking_metric", "RankingMetric"]

import typing as t
from dataclasses import dataclass

from pydantic import Field, create_model

from . import Metric
from .decorator import create_metric_decorator


@dataclass
class RankingMetric(Metric):
    num_ranks: int = 2

    def __post_init__(self):
        super().__post_init__()
        self._response_model = create_model(
            "RankingResponseModel",
            result=(t.List[str], Field(..., description="List of ranked items")),
            reason=(str, Field(..., description="Reasoning for the ranking")),
        )


ranking_metric = create_metric_decorator(RankingMetric)
