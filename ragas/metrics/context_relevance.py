from __future__ import annotations

import typing as t
from dataclasses import dataclass

import torch
from datasets import Dataset
from pandas import DataFrame

from ragas.metrics.answer_relevance import QGen
from ragas.metrics.base import Metric


@dataclass
class ContextRelevancy(Metric):
    batch_size: int = 32
    name: str = "context_relavency"
    model_name: str = "t5-base"

    def init_model(self: t.Self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = QGen(self.model_name, self.device)

    @staticmethod
    def _make_question_context_rank_pairs(row: dict) -> dict:
        q_id = row["q_id"]
        q = row["question"]
        cs = row["contexts"]
        new_q_ids = []
        sentences = []
        for i, q_id in enumerate(q_id):
            for c in cs[i]:
                sentences.append([q[i], c])
                new_q_ids.append(q_id)

        return {"q_id": new_q_ids, "sentences": sentences}

    def score(self: t.Self, dataset: Dataset) -> Dataset:
        """
        dataset: Dataset[question: list[str], contexts: list[list[str]]]
        """

        num_qs, _ = dataset.shape
        dataset = dataset.add_column("q_id", list(range(num_qs)))  # type: ignore
        sentence_ds = dataset.map(
            self._make_question_context_rank_pairs,
            batched=True,
            batch_size=10,
            remove_columns=dataset.column_names,
        )

        # we loose memory here because we have to make it py_list
        scores = self.model.predict(sentence_ds["sentences"])

        sentence_ds = sentence_ds.add_column(self.name, scores)  # type: ignore
        df = sentence_ds.to_pandas()
        assert isinstance(df, DataFrame)
        score_ds = Dataset.from_dict({self.name: df.groupby("q_id")[self.name].mean()})

        return score_ds


context_relevancy = ContextRelevancy()
