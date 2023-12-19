from __future__ import annotations

import typing as t
from dataclasses import dataclass

import numpy as np
from datasets import Dataset
from langchain.callbacks.manager import CallbackManager, trace_as_chain_group
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from ragas.llms.prompt import Prompt
from ragas.metrics.base import EvaluationMode, MetricWithLLM
from ragas.utils import json_loader

if t.TYPE_CHECKING:
    from langchain.callbacks.base import Callbacks

CONTEXT_PRECISION = HumanMessagePromptTemplate.from_template(
    """\
Verify if the information in the given context is useful in answering the question.

question: What are the health benefits of green tea?
context: 
This article explores the rich history of tea cultivation in China, tracing its roots back to the ancient dynasties. It discusses how different regions have developed their unique tea varieties and brewing techniques. The article also delves into the cultural significance of tea in Chinese society and how it has become a symbol of hospitality and relaxation.
verification:
{{"reason":"The context, while informative about the history and cultural significance of tea in China, does not provide specific information about the health benefits of green tea. Thus, it is not useful for answering the question about health benefits.", "verdict":"No"}}

question: How does photosynthesis work in plants?
context:
Photosynthesis in plants is a complex process involving multiple steps. This paper details how chlorophyll within the chloroplasts absorbs sunlight, which then drives the chemical reaction converting carbon dioxide and water into glucose and oxygen. It explains the role of light and dark reactions and how ATP and NADPH are produced during these processes.
verification:
{{"reason":"This context is extremely relevant and useful for answering the question. It directly addresses the mechanisms of photosynthesis, explaining the key components and processes involved.", "verdict":"Yes"}}

question:{question}
context:
{context}
verification:"""  # noqa: E501
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

    name: str = "context_precision"  # type: ignore
    evaluation_mode: EvaluationMode = EvaluationMode.qc  # type: ignore
    batch_size: int = 15

    def _context_precision_prompt(self, row: t.Dict) -> t.List[Prompt]:
        question, contexts = row["question"], row["contexts"]

        human_prompts = [
            ChatPromptTemplate.from_messages(
                [CONTEXT_PRECISION.format(question=question, context=c)]
            )
            for c in contexts
        ]
        return [Prompt(chat_prompt_template=hp) for hp in human_prompts]

    def _calculate_average_precision(self, responses: t.List[str]) -> float:
        score = np.nan
        response = [json_loader.safe_load(item, self.llm) for item in responses]
        response = [
            int("yes" in resp.get("verdict", " ").lower())
            if resp.get("verdict")
            else np.nan
            for resp in response
        ]
        denominator = sum(response) + 1e-10
        numerator = sum(
            [
                (sum(response[: i + 1]) / (i + 1)) * response[i]
                for i in range(len(response))
            ]
        )
        score = numerator / denominator
        return score

    async def _ascore(
        self: t.Self,
        row: t.Dict,
        callbacks: Callbacks = [],
    ) -> float:
        assert self.llm is not None, "LLM is not set"

        human_prompts = self._context_precision_prompt(row)
        responses: t.List[str] = []
        for hp in human_prompts:
            result = await self.llm.agenerate_text(
                hp,
                n=1,
                callbacks=callbacks,
            )
            responses.append(result.generations[0][0].text)

        score = self._calculate_average_precision(responses)
        return score

    def _score(self, row: t.Dict, callbacks: Callbacks = []) -> float:
        assert self.llm is not None, "LLM is not set"

        human_prompts = self._context_precision_prompt(row)
        responses: t.List[str] = []
        for hp in human_prompts:
            result = self.llm.generate_text(
                hp,
                n=1,
                callbacks=callbacks,
            )
            responses.append(result.generations[0][0].text)

        score = self._calculate_average_precision(responses)
        return score


context_precision = ContextPrecision()
