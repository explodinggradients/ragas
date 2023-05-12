from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

from Levenshtein import distance, ratio
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

from belar.metrics.base import Metric

ROUGE_TYPES = t.Literal["rouge1", "rouge2", "rougeL"]


@dataclass
class BLEU(Metric):
    weights: list[float] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])
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
        return [
            sentence_bleu(
                s1,
                s2,
                weights=self.weights,
                smoothing_function=self.smoothing_function,
            )
            for s1, s2 in zip(ground_truth_, generated_text_)
        ]


@dataclass
class ROUGE(Metric):
    type: t.Literal[ROUGE_TYPES]
    use_stemmer: bool = False

    def __post_init__(self):
        self.scorer = rouge_scorer.RougeScorer(
            [self.type], use_stemmer=self.use_stemmer
        )

    @property
    def name(self):
        return self.type

    @property
    def is_batchable(self):
        return False

    def score(
        self, ground_truths: list[str], generated_texts: list[str]
    ) -> list[float]:
        scores = [
            self.scorer.score(ground_truth, generated_texts[i])[self.type].fmeasure
            for i, ground_truth in enumerate(ground_truths)
        ]
        return scores


@dataclass
class EditScore(Metric):
    measure: t.Literal["distance", "ratio"] = "ratio"

    @property
    def name(self) -> str:
        return f"edit_{self.measure}"

    @property
    def is_batchable(self):
        return True

    def score(self, ground_truth: t.List[str], generated_text: t.List[str]):
        if self.measure == "distance":
            score = [distance(s1, s2) for s1, s2 in zip(ground_truth, generated_text)]
        elif self.measure == "ratio":
            score = [ratio(s1, s2) for s1, s2 in zip(ground_truth, generated_text)]
        else:
            raise ValueError(f"Unkown measure {self.measure}")

        return score


Rouge1 = ROUGE("rouge1")
Rouge2 = ROUGE("rouge2")
RougeL = ROUGE("rougeL")
BLUE = BLEU()
EditDistance = EditScore("distance")
EditRatio = EditScore("ratio")

__all__ = ["Rouge1", "Rouge2", "RougeL", "BLEU", "EditDistance", "EditRatio"]
