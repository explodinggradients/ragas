import typing as t
from dataclasses import dataclass, field

from langchain_core.callbacks import Callbacks

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._faithfulness import HasSegmentMethod
from ragas.metrics.base import MetricType, SingleTurnMetric, get_segmenter
from ragas.run_config import RunConfig


@dataclass
class BleuScore(SingleTurnMetric):
    name: str = "bleu_score"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {MetricType.SINGLE_TURN: {"reference", "response"}}
    )
    weights: t.Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    sentence_segmenter: t.Optional[HasSegmentMethod] = None
    language: str = "english"

    def __post_init__(self):
        try:
            from nltk.tokenize import word_tokenize
            from nltk.translate.bleu_score import corpus_bleu
        except ImportError:
            raise ImportError(
                "nltk is required for bleu score. Please install it using `pip install nltk`"
            )
        if not self.sentence_segmenter:
            self.sentence_segmenter = get_segmenter(language=self.language, clean=False)
        self.word_tokenizer = word_tokenize
        self.corpus_bleu = corpus_bleu

    def init(self, run_config: RunConfig):
        pass

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:

        assert (
            self.sentence_segmenter is not None
        ), "Sentence segmenter is not initialized"

        reference_sentences = self.sentence_segmenter.segment(sample.reference)
        response_sentences = self.sentence_segmenter.segment(sample.response)

        reference = [
            [self.word_tokenizer(reference)] for reference in reference_sentences
        ]
        response = [self.word_tokenizer(response) for response in response_sentences]
        score = self.corpus_bleu(reference, response, weights=self.weights)
        assert isinstance(score, float), "Expecting a float"
        return score

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)


bleu_score = BleuScore()
