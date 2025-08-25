import typing as t
from dataclasses import dataclass, field

from langchain_core.callbacks import Callbacks

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import MetricType, SingleTurnMetric
from ragas.run_config import RunConfig


@dataclass
class ChrfScore(SingleTurnMetric):
    name: str = "chrf_score"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {MetricType.SINGLE_TURN: {"reference", "response"}}
    )
    language: str = "english"
    kwargs: t.Dict[str, t.Any] = field(default_factory=dict)

    def __post_init__(self):
        try:
            from sacrebleu import corpus_chrf
        except ImportError:
            raise ImportError(
                "sacrebleu is required for chrf score. Please install it using `pip install sacrebleu`"
            )
        self.corpus_chrf = corpus_chrf

    def init(self, run_config: RunConfig):
        pass

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        reference, response = sample.reference, sample.response
        assert isinstance(reference, str), "ChrfScore expects a valid reference string"
        assert isinstance(response, str), "ChrfScore expects a valid response string"

        reference_sentences = reference.split(". ")
        response_sentences = response.split(". ")

        # corpus_chrf expects a list of strings and a list of list of strings
        references = [[ref] for ref in reference_sentences]
        hypotheses = response_sentences

        score = self.corpus_chrf(hypotheses, references, **self.kwargs).score / 100
        assert isinstance(score, float), "Expecting a float"
        return score

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)
