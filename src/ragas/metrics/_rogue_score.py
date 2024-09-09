import typing as t
from dataclasses import dataclass, field

from langchain_core.callbacks import Callbacks
from rouge_score import rouge_scorer

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import MetricType, SingleTurnMetric
from ragas.run_config import RunConfig


@dataclass
class RougeScore(SingleTurnMetric):
    name: str = "rouge_score"  # type: ignore
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {MetricType.SINGLE_TURN: {"reference", "response"}}
    )
    rogue_type: t.Literal["rouge1", "rougeL"] = "rougeL"
    measure_type: t.Literal["fmeasure", "precision", "recall"] = "fmeasure"

    def init(self, run_config: RunConfig):
        pass

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        assert isinstance(sample.reference, str), "Sample reference must be a string"
        assert isinstance(sample.response, str), "Sample response must be a string"
        scorer = rouge_scorer.RougeScorer([self.rogue_type], use_stemmer=True)
        scores = scorer.score(sample.reference, sample.response)
        return getattr(scores[self.rogue_type], self.measure_type)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)


rouge_score = RougeScore()
