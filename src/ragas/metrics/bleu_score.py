import typing as t
from dataclasses import dataclass, field

from langchain_core.callbacks import Callbacks
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._faithfulness import HasSegmentMethod
from ragas.metrics.base import MetricType, SingleTurnMetric, get_segmenter
from ragas.run_config import RunConfig


@dataclass
class BleuScore(SingleTurnMetric):
    name: str = "bleu_score"  # type: ignore
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {MetricType.SINGLE_TURN: {"reference", "response"}}
    )
    weights: t.Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    sentence_segmenter: t.Optional[HasSegmentMethod] = None

    def __post_init__(self):
        self.segmenter = get_segmenter()

    def init(self, run_config: RunConfig):
        pass

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        reference_sentences = self.segmenter.segment(sample.reference)
        response_sentences = self.segmenter.segment(sample.response)

        reference = [[word_tokenize(reference)] for reference in reference_sentences]
        response = [word_tokenize(response) for response in response_sentences]
        score = corpus_bleu(reference, response, weights=self.weights)
        assert isinstance(score, float), "Expecting a float"
        return score

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)


bleu_score = BleuScore()
