import typing as t
from dataclasses import dataclass

from langchain_core.callbacks import Callbacks
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._faithfulness import HasSegmentMethod
from ragas.metrics.base import SingleTurnMetric, get_segmenter
from ragas.run_config import RunConfig


@dataclass
class BleuScore(SingleTurnMetric):
    name: str = "bleu_score"  # type: ignore
    _required_columns: t.Tuple[str, ...] = ("reference", "response")
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


@dataclass
class RougeScore(SingleTurnMetric):
    name: str = "rouge_score"  # type: ignore
    _required_columns: t.Tuple[str, ...] = ("reference", "response")
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


bleu_score = BleuScore()
