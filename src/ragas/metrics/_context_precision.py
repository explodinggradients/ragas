from __future__ import annotations

import typing as t
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pysbd
from datasets import Dataset
from langchain.callbacks.manager import CallbackManager, trace_as_chain_group
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate

from ragas.metrics.base import EvaluationMode, MetricWithLLM

CONTEXT_PRECISION = HumanMessagePromptTemplate.from_template(
    """\
Given a question and a context, verify if the information in the given context is useful in answering the question. Return a Yes/No answer.
question:{question}
context:\n{context}
answer:
"""  # noqa: E501
)

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
    human_template: HumanMessagePromptTemplate = field(default_factory=lambda: CONTEXT_PRECISION)
    ai_template: AIMessagePromptTemplate = None

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
                for c in ctx:
                    messages = [self.human_template.format(question=qstn, context=c)]
                    if self.ai_template is not None:
                        messages.append(self.ai_template.format())
                    prompts.append(ChatPromptTemplate.from_messages(messages))
            self.logs["prompts"] += prompts

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
            self.logs["grouped_responses"] += grouped_responses

            scores = []
            for response in grouped_responses:
                response = [int(any("yes" in r.strip().lower() for r in resp)) for resp in response]
                self.logs["responses"].append(response)
                denominator = sum(response) if sum(response) != 0 else 1e-10
                self.logs["denominator"].append(denominator)
                numerator = sum(
                    [
                        (sum(response[: i + 1]) / (i + 1)) * response[i]
                        for i in range(len(response))
                    ]
                )
                self.logs["numerator"].append(numerator)
                scores.append(numerator / denominator)

        self.logs["scores"] += scores
        return scores


context_precision = ContextPrecision()