from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass
from typing import List

import numpy as np
import pysbd
from datasets import Dataset
from langchain.callbacks.manager import CallbackManager, trace_as_chain_group
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from ragas.metrics.base import EvaluationMode, MetricWithLLM

CONTEXT_RELEVANCE = HumanMessagePromptTemplate.from_template(
    """\
Please extract relevant sentences from the provided context that is absolutely required answer the following question. If no relevant sentences are found, or if you believe the question cannot be answered from the given context, return the phrase "Insufficient Information".  While extracting candidate sentences you're not allowed to make any changes to sentences from given context.

question:{question}
context:\n{context}
candidate sentences:\n"""  # noqa: E501
)

CONTEXT_PRECISION = HumanMessagePromptTemplate.from_template(
    """\
Given a question and a context, verify if the information in the given context is useful in answering the question. Return a Yes/No answer.
question:{question}
context:\n{context}
answer:
"""  # noqa: E501
)


seg = pysbd.Segmenter(language="en", clean=False)


def sent_tokenize(text: str) -> List[str]:
    """
    tokenizer text into sentences
    """
    sentences = seg.segment(text)
    assert isinstance(sentences, list)
    return sentences


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
    """

    name: str = "context_relevancy"
    evaluation_mode: EvaluationMode = EvaluationMode.qc
    batch_size: int = 15
    show_deprecation_warning: bool = False

    def __post_init__(self: t.Self):
        pass

    def _score_batch(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[CallbackManager] = None,
        callback_group_name: str = "batch",
    ) -> list[float]:
        if self.show_deprecation_warning:
            logging.warning(
                "The 'context_relevancy' metric is going to be deprecated soon! Please use the 'context_precision' metric instead. It is a drop-in replacement just a simple search and replace should work."  # noqa
            )
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
            results = self.llm.generate(
                prompts,
                n=1,
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
                        sent_tokenize(output.strip())
                        if output.lower() != "insufficient information."
                        else []
                    )
                    if len(context_sents) == 0:
                        score = 0
                    else:
                        score = min(len(indices) / len(context_sents), 1)
                    overlap_scores.append(score)
                scores.append(np.mean(overlap_scores))

        return scores


@dataclass
class ContextPrecision(MetricWithLLM):
    """
    Average Precision is a metric that evaluates whether all of the
    relevant items selected by the model are ranked higher or not.

    Attributes
    ----------
    name : str
    batch_size : int
        Batch size for openai completion.
    """

    name: str = "context_precision"
    evaluation_mode: EvaluationMode = EvaluationMode.qc
    batch_size: int = 15

    def _score_batch(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[CallbackManager] = None,
        callback_group_name: str = "batch",
    ) -> list:
        prompts = []
        questions, contexts = dataset["question"], dataset["contexts"]
        with trace_as_chain_group(
            callback_group_name, callback_manager=callbacks
        ) as batch_group:
            for qstn, ctx in zip(questions, contexts):
                human_prompts = [
                    ChatPromptTemplate.from_messages(
                        [CONTEXT_PRECISION.format(question=qstn, context=c)]
                    )
                    for c in ctx
                ]

                prompts.extend(human_prompts)

            responses: list[list[str]] = []
            results = self.llm.generate(
                prompts,
                n=1,
                callbacks=batch_group,
            )
            responses = [[i.text for i in r] for r in results.generations]
            context_lens = [len(ctx) for ctx in contexts]
            context_lens.insert(0, 0)
            context_lens = np.cumsum(context_lens)
            grouped_responses = [
                responses[start:end]
                for start, end in zip(context_lens[:-1], context_lens[1:])
            ]
            scores = []

            for response in grouped_responses:
                response = [int("Yes" in resp) for resp in response]
                denominator = sum(response) + 1e-10
                numerator = sum(
                    [
                        (sum(response[: i + 1]) / (i + 1)) * response[i]
                        for i in range(len(response))
                    ]
                )
                scores.append(numerator / denominator)

        return scores


context_precision = ContextPrecision()
context_relevancy = ContextRelevancy()
