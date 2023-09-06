from __future__ import annotations

import typing as t
from dataclasses import dataclass
from itertools import combinations, product
from typing import List

import numpy as np
from datasets import Dataset
from langchain.callbacks.manager import CallbackManager, trace_as_chain_group
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from sentence_transformers import CrossEncoder

from ragas.metrics.base import EvaluationMode, MetricWithLLM
from ragas.metrics.llms import generate

CONTEXT_RELEVANCE = HumanMessagePromptTemplate.from_template(
    """\
Please extract relevant sentences from the provided context that can potentially help answer the following question. If no relevant sentences are found, or if you believe the question cannot be answered from the given context, return the phrase "Insufficient Information".  While extracting candidate sentences you're not allowed to make any changes to sentences from given context.

question:{question}
context:\n{context}
candidate sentences:\n"""  # noqa: E501
)


def sent_tokenize(sent: str) -> List[str]:
    return [s[:-1] if s.endswith(".") else s for s in sent.strip().split(". ")]


class SentenceAgreement:
    def __init__(
        self: t.Self,
        model_name: str = "cross-encoder/stsb-TinyBERT-L-4",
        metric: str = "bert_score",
    ):
        self.metric = metric
        self.cross_encoder = CrossEncoder(model_name)

    def bert_score(self, para1: str, para2: str) -> float:
        sentences1, sentences2 = sent_tokenize(para1), sent_tokenize(para2)
        scores = self.cross_encoder.predict(
            list(product(sentences1, sentences2)), convert_to_numpy=True  # type: ignore
        )
        assert isinstance(scores, np.ndarray), "Expects ndarray"
        scores = scores.reshape(len(sentences1), len(sentences2))
        return scores.max(axis=1).mean()

    @staticmethod
    def jaccard_score(para1: str, para2: str) -> float:
        sentences1, sentences2 = sent_tokenize(para1), sent_tokenize(para2)
        intersect = len(np.intersect1d(sentences1, sentences2))
        union = len(np.union1d(sentences1, sentences2))
        return intersect / union

    def evaluate(self, answers: List[str]) -> np.float_:
        """
        eval nC2 combinations
        """
        scores = []
        groups = combinations(answers, 2)
        for group in groups:
            if self.metric == "jaccard":
                score = self.jaccard_score(*group)  # type: ignore
            elif self.metric == "bert_score":
                score = self.bert_score(*group)  # type: ignore
            else:
                score = 0
                raise ValueError(f"Metric {self.metric} unavailable")
            scores.append(score)
        score = np.mean(scores)
        return score


@dataclass
class ContextRelevancy(MetricWithLLM):
    """
    Extracts sentences from the context that are relevant to the question with
    self-consistancy checks. The number of relevant sentences and is used as the score.

    Attributes
    ----------
    name : str
    batch_size : int
        Batch size for openai completion.
    strictness : int
        Controls the number of times sentence extraction is performed to quantify
        uncertainty from the LLM. Defaults to 2.
    agreement_metric : str
        "bert_score" or "jaccard_score", used to measure agreement between multiple
        samples.
    model_name : str
        any encoder model. Used for calculating bert_score.
    """

    name: str = "context_relevancy"
    evaluation_mode: EvaluationMode = EvaluationMode.qc
    batch_size: int = 15
    strictness: int = 2
    agreement_metric: str = "bert_score"
    model_name: str = "cross-encoder/stsb-TinyBERT-L-4"

    def __post_init__(self: t.Self):
        if self.agreement_metric == "bert_score" and self.model_name is None:
            raise ValueError(
                "model_name must be provided when agreement_metric is bert_score"
            )
        self.temperature = 0.2 if self.strictness > 0 else 0

    def init_model(self: t.Self):
        self.sent_agreement = SentenceAgreement(
            model_name=self.model_name, metric=self.agreement_metric
        )

    def _score_batch(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[CallbackManager] = None,
        callback_group_name: str = "batch",
    ) -> list[float]:
        prompts = []
        questions, contexts = dataset["question"], dataset["contexts"]
        with trace_as_chain_group(
            callback_group_name, callback_manager=callbacks
        ) as batch_group:
            for q, c in zip(questions, contexts):
                human_prompt = CONTEXT_RELEVANCE.format(
                    question=q, context="\n".join(c)
                )
                prompts.append(ChatPromptTemplate.from_messages([human_prompt]))

            responses: list[list[str]] = []
            results = generate(
                prompts,
                self.llm,
                n=self.strictness,
                temperature=self.temperature,
                callbacks=batch_group,
            )
            responses = [[i.text for i in r] for r in results.generations]

            scores = []
            for context, n_response in zip(contexts, responses):
                context = "\n".join(context)
                overlap_scores = []
                context_sents = sent_tokenize(context)
                for output in n_response:
                    indices = (
                        output.split("\n")
                        if output.lower() != "insufficient information."
                        else []
                    )
                    score = min(len(indices) / len(context_sents), 1)
                    overlap_scores.append(score)
                if self.strictness > 1:
                    agr_score = self.sent_agreement.evaluate(n_response)
                else:
                    agr_score = 1
                scores.append(agr_score * np.mean(overlap_scores))

        return scores


context_relevancy = ContextRelevancy()
