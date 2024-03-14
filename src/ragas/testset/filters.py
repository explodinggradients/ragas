from __future__ import annotations

import logging
import typing as t
from abc import ABC
from dataclasses import dataclass, field


from ragas.llms.json_load import json_loader
from ragas.run_config import RunConfig
from ragas.testset.prompts import (
    context_scoring_prompt,
    evolution_elimination_prompt,
    filter_question_prompt,
)

if t.TYPE_CHECKING:
    from ragas.llms.base import BaseRagasLLM
    from ragas.llms.prompt import Prompt
    from ragas.testset.docstore import Node


logger = logging.getLogger(__name__)


@dataclass
class Filter(ABC):
    llm: BaseRagasLLM

    def set_run_config(self, run_config: RunConfig):
        self.llm.set_run_config(run_config)

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        """
        Adapt the filter to a different language.
        """
        raise NotImplementedError("adapt() is not implemented for {} Filter")

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        """
        Save the filter prompts to a path.
        """
        raise NotImplementedError("save() is not implemented for {} Filter")


@dataclass
class NodeFilter(Filter):
    threshold: float = 7.5
    context_scoring_prompt: Prompt = field(
        default_factory=lambda: context_scoring_prompt
    )

    async def filter(self, node: Node) -> t.Dict:
        prompt = self.context_scoring_prompt.format(context=node.page_content)
        results = await self.llm.generate(prompt=prompt)
        output = results.generations[0][0].text.strip()
        score = await json_loader.safe_load(output, llm=self.llm)
        score_dict = score if isinstance(score, dict) else {}
        logger.debug("node filter: %s", score)
        score = score_dict.get("score", 0)
        try:
            score = float(score)
        except Exception as _:
            score = 0
        score_dict.update({"score": score >= self.threshold})
        return score_dict

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        """
        Adapt the filter to a different language.
        """
        self.context_scoring_prompt = self.context_scoring_prompt.adapt(
            language, self.llm, cache_dir
        )

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        """
        Save the filter prompts to a path.
        """
        self.context_scoring_prompt.save(cache_dir)


@dataclass
class QuestionFilter(Filter):
    llm: BaseRagasLLM
    filter_question_prompt: Prompt = field(
        default_factory=lambda: filter_question_prompt
    )

    async def filter(self, question: str) -> t.Tuple[bool, str]:
        prompt = self.filter_question_prompt.format(question=question)
        results = await self.llm.generate(prompt=prompt)
        results = results.generations[0][0].text.strip()
        json_results = await json_loader.safe_load(results, llm=self.llm)
        json_results = json_results if isinstance(json_results, dict) else {}
        logger.debug("filtered question: %s", json_results)
        return json_results.get("verdict") == "1", json_results.get("feedback", "")

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        """
        Adapt the filter to a different language.
        """
        self.filter_question_prompt = self.filter_question_prompt.adapt(
            language, self.llm, cache_dir
        )

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        """
        Save the filter prompts to a path.
        """
        self.filter_question_prompt.save(cache_dir)


@dataclass
class EvolutionFilter(Filter):
    llm: BaseRagasLLM
    evolution_elimination_prompt: Prompt = field(
        default_factory=lambda: evolution_elimination_prompt
    )

    async def filter(self, simple_question: str, compressed_question: str) -> bool:
        prompt = self.evolution_elimination_prompt.format(
            question1=simple_question, question2=compressed_question
        )
        results = await self.llm.generate(prompt=prompt)
        results = results.generations[0][0].text.strip()
        json_results = await json_loader.safe_load(results, llm=self.llm)
        json_results = json_results if isinstance(json_results, dict) else {}
        logger.debug("evolution filter: %s", json_results)
        return json_results.get("verdict") == "1"

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        """
        Adapt the filter to a different language.
        """
        self.evolution_elimination_prompt = self.evolution_elimination_prompt.adapt(
            language, self.llm, cache_dir
        )

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        """
        Save the filter prompts to a path.
        """
        self.evolution_elimination_prompt.save(cache_dir)
