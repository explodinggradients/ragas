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
from tqdm import tqdm

from ragas.metrics.base import EvaluationMode, MetricWithLLM
from ragas.metrics.llms import generate

CONTEXT_RELEVANCE = HumanMessagePromptTemplate.from_template(
    """\
Task: Candidate sentence extraction.
Given the question and context, extract minimum number of sentences from context required to answer the question. If the context do not contain information required to answer the question return "No candidate sentences found".

question: Which equation is known as worlds most famous equation?
context:\nAlbert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist,[5] widely ranked among the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century.
His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been called "the world's most famous equation".
sentences:His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been called "the world's most famous equation".

question: Were Scott Derrickson and Ed Wood of the same nationality?
context :\nScott Derrickson (born July 16, 1966) is an American director, screenwriter and producer He lives in Los Angeles, California He is best known for directing horror films such as "Sinister", "The Exorcism of Emily Rose", and "Deliver Us From Evil", as well as the 2016 Marvel Cinematic Universe installment, "Doctor Strange"Tyler Bates is an American musician, music producer, and composer for films, television, and video games. Adam Collis is an American filmmaker and actor.Conrad Brooks is an American actor.Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker, actor, writer, producer, and director.
sentences:Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer. Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker, actor, writer, producer, and director.

question: How many were killed in the Tiananmen Square incident?
context:\nTiananmen Square incident, also called June Fourth incident or 6/4, series of protests and demonstrations in China in the spring of 1989 that culminated on the night of June 3–4 with a government crackdown on the demonstrators in Tiananmen Square in Beijing.
sentences: No candidate sentences found.

question:{question}
context:\n{context}
sentences:"""  # noqa: E501
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

    name: str = "context_ relevancy"
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

    def score(self: t.Self, dataset: Dataset) -> Dataset:
        """
        Parameters
        ----------
        dataset: Dataset[question: list[str], contexts: list[list[str]]]

        Returns
        -------
        Dataset[question: list[str], contexts: list[list[str]], scores: list[float]]
            Dataset with the scores for each row.
        """
        if self.llm is None:
            raise ValueError("llm must not be None")

        scores = []
        with trace_as_chain_group(f"ragas_{self.name}") as score_group:
            for batch in tqdm(self.get_batches(len(dataset))):
                score = self._score_batch(dataset.select(batch), callbacks=score_group)
                scores.extend(score)

        return dataset.add_column(self.name, scores)  # type: ignore

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
                    indices = sent_tokenize(output)
                    overlap_scores.append(len(indices) / len(context_sents))
                if self.strictness > 1:
                    agr_score = self.sent_agreement.evaluate(n_response)
                else:
                    agr_score = 1
                scores.append(agr_score * np.mean(overlap_scores))

        return scores


context_relevancy = ContextRelevancy()
