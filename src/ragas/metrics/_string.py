import typing as t
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.callbacks import Callbacks
from rapidfuzz import distance

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import MetricType, SingleTurnMetric
from ragas.run_config import RunConfig


class DistanceMeasure(Enum):
    LEVENSHTEIN = "levenshtein"
    HAMMING = "hamming"
    JARO = "jaro"


DISTANCE_MEASURE_MAP = {
    DistanceMeasure.LEVENSHTEIN: distance.Levenshtein,
    DistanceMeasure.HAMMING: distance.Hamming,
    DistanceMeasure.JARO: distance.Jaro,
}


@dataclass
class ExactMatch(SingleTurnMetric):
    name: str = "exact_match"  # type: ignore
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {MetricType.SINGLE_TURN: {"reference", "response"}}
    )

    def init(self, run_config: RunConfig):
        pass

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        return float(sample.reference == sample.response)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)


class StringPresent(SingleTurnMetric):
    name: str = "string_present"  # type: ignore
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {MetricType.SINGLE_TURN: {"reference", "response"}}
    )

    def init(self, run_config: RunConfig):
        pass

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        reference = sample.reference
        response = sample.response
        assert isinstance(reference, str), "Expecting a string"
        assert isinstance(response, str), "Expecting a string"
        return float(reference in response)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)


class StringDistance(SingleTurnMetric):
    name: str = "string_distance"  # type: ignore
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {MetricType.SINGLE_TURN: {"reference", "response"}}
    )
    distance_measure: DistanceMeasure = DistanceMeasure.LEVENSHTEIN

    def init(self, run_config: RunConfig):
        pass

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        reference = sample.reference
        response = sample.response
        assert isinstance(reference, str), "Expecting a string"
        assert isinstance(response, str), "Expecting a string"
        return DISTANCE_MEASURE_MAP[self.distance_measure].distance(reference, response)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)
