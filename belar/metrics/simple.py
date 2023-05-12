from __future__ import annotations

import typing as t
from dataclasses import dataclass
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer

from belar.metrics.base import Metric

ROUGE_TYPES = t.Literal["rouge1", "rouge2", "rougeL"]


@dataclass
class ROUGE(Metric):
    type: t.Literal[ROUGE_TYPES]
    use_stemmer: bool = False

    def __post_init__(self):
        self.scorer = rouge_scorer.RougeScorer(
            [self.type], use_stemmer=self.use_stemmer
        )

    def name(self):
        return self.type

    def is_batchable(self):
        return False

    def score(self, ground_truth: str, generated_text: str):
        if isinstance(ground_truth, list):
            ground_truth = ground_truth[0]
        if isinstance(generated_text, list):
            generated_text = generated_text[0]

        score = self.scorer.score(ground_truth, generated_text)[self.type]
        return score.fmeasure


class BLEU(Metric):
    weights: t.List[float] = [0.25, 0.25, 0.25, 0.25]
    smoothing_function = None

    @property
    def name(self):
        return "BLEU"

    @property
    def is_batchable(self):
        return True

    def score(self, ground_truth: t.List[str], generated_text: t.List[str]):
        ground_truth_ = [[word_tokenize(text)] for text in ground_truth]
        generated_text_ = [word_tokenize(text) for text in generated_text]
        return corpus_bleu(
            ground_truth_,
            generated_text_,
            weights=self.weights,
            smoothing_function=self.smoothing_function,
        )


Rouge1 = ROUGE("rouge1")
Rouge2 = ROUGE("rouge2")
RougeL = ROUGE("rougeL")

__all__ = ["Rouge1", "Rouge2", "RougeL"]
