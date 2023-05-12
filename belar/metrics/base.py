from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from datasets import Dataset


@dataclass
class Metric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def is_batchable(self) -> bool:
        ...

    @abstractmethod
    def score(self, ground_truth: list[str], generated_text: list[str]) -> list[float]:
        ...


@dataclass
class Evaluation:
    metrics: list[Metric]
    batched: bool = True
    batch_size: int = 1000

    def eval(self, ground_truth: list[list[str]], generated_text: list[str]) -> Dataset:
        ds = Dataset.from_dict(
            {"ground_truth": ground_truth, "generated_text": generated_text}
        )
        ds = ds.map(self._get_score, batched=self.batched, batch_size=self.batch_size)

        return ds

    # TODO: set a typevar here for row
    def _get_score(self, row: dict[str, list[t.Any]] | dict[str, t.Any]):
        for metric in self.metrics:
            if self.batched:
                split_indices = []
                last_split_index = 0
                ground_truths = []
                generated_texts = []
                for i, ground_truth_list in enumerate(row["ground_truth"]):
                    split_indices.append(last_split_index + len(ground_truth_list))
                    last_split_index = split_indices[-1]
                    ground_truths.extend(ground_truth_list)
                    generated_texts.extend(
                        [row["generated_text"][i]] * len(ground_truth_list)
                    )

                # contruct variable array back and compute score
                batch_scores_flat = metric.score(ground_truths, generated_texts)
                batch_scores = np.split(batch_scores_flat, split_indices)
                score = [np.max(x) for x in batch_scores[:-1]]
            else:  # not batched
                split_indices = len(row["ground_truth"])
                ground_truths = row["ground_truth"]
                generated_texts = [row["generated_text"]] * split_indices
                scores = metric.score(ground_truths, generated_texts)
                score = np.max(scores)

            row[f"{metric.name}_score"] = score

        return row
