from __future__ import annotations

import logging
import typing as t
from abc import ABC
from dataclasses import dataclass

from ragas.llms.json_load import load_as_json
from ragas.testset.prompts import (
    context_scoring_prompt,
    evolution_elimination_prompt,
    filter_question_prompt,
)

if t.TYPE_CHECKING:
    from ragas.llms.base import BaseRagasLLM
    from ragas.testset.docstore import Node


logger = logging.getLogger(__name__)


@dataclass
class Filter(ABC):
    ...


@dataclass
class NodeFilter(Filter):
    llm: BaseRagasLLM
    threshold: float = 7.5

    async def filter(self, node: Node) -> t.Dict:
        prompt = context_scoring_prompt.format(context=node.page_content)
        results = await self.llm.agenerate_text(prompt=prompt)
        output = results.generations[0][0].text.strip()
        score = load_as_json(output)
        score.update({"score": score.get("score", 0) >= self.threshold})
        return score


@dataclass
class QuestionFilter(Filter):
    llm: BaseRagasLLM

    async def filter(self, question: str) -> bool:
        prompt = filter_question_prompt.format(question=question)
        results = await self.llm.agenerate_text(prompt=prompt)
        results = results.generations[0][0].text.strip()
        json_results = load_as_json(results)
        logger.debug("filtered question: %s", json_results)
        return json_results.get("verdict") != "No"


@dataclass
class EvolutionFilter(Filter):
    llm: BaseRagasLLM

    async def filter(self, simple_question: str, compressed_question: str) -> bool:
        prompt = evolution_elimination_prompt.format(
            question1=simple_question, question2=compressed_question
        )
        results = await self.llm.agenerate_text(prompt=prompt)
        results = results.generations[0][0].text.strip()
        json_results = load_as_json(results)
        logger.debug("filtered question: %s", json_results)
        return json_results.get("verdict") != "No"
