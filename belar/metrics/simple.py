from __future__ import annotations

from Levenshtein import distance, ratio
import typing as t
from dataclasses import dataclass

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

    @property
    def name(self):
        return self.type

    @property
    def is_batchable(self):
        return False

    def score(self, ground_truth: str, generated_text: str):
        if isinstance(ground_truth, list):
            ground_truth = ground_truth[0]
        if isinstance(generated_text, list):
            generated_text = generated_text[0]

        score = self.scorer.score(ground_truth, generated_text)[self.type]
        return score.fmeasure


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

__all__ = ["Rouge1", "Rouge2", "RougeL"]
