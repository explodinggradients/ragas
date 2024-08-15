from __future__ import annotations

import logging
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ragas.llms.json_load import json_loader
from ragas.testset.prompts import keyphrase_extraction_prompt

if t.TYPE_CHECKING:
    from ragas.llms.base import BaseRagasLLM
    from ragas.llms.prompt import Prompt
    from ragas.testset.docstore import Node


logger = logging.getLogger(__name__)


@dataclass
class Extractor(ABC):
    llm: BaseRagasLLM

    @abstractmethod
    async def extract(self, node: Node, is_async: bool = True) -> t.Any:
        ...

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        """
        Adapt the extractor to a different language.
        """
        raise NotImplementedError("adapt() is not implemented for {} Extractor")

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        """
        Save the extractor prompts to a path.
        """
        raise NotImplementedError("adapt() is not implemented for {} Extractor")


@dataclass
class KeyphraseExtractor(Extractor):
    extractor_prompt: Prompt = field(
        default_factory=lambda: keyphrase_extraction_prompt
    )

    async def extract(self, node: Node, is_async: bool = True) -> t.List[str]:
        prompt = self.extractor_prompt.format(text=node.page_content)
        results = await self.llm.generate(prompt=prompt, is_async=is_async)
        keyphrases = await json_loader.safe_load(
            results.generations[0][0].text.strip(), llm=self.llm, is_async=is_async
        )
        keyphrases = keyphrases if isinstance(keyphrases, dict) else {}
        logger.debug("topics: %s", keyphrases)
        return keyphrases.get("keyphrases", [])

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        """
        Adapt the extractor to a different language.
        """
        self.extractor_prompt = self.extractor_prompt.adapt(
            language, self.llm, cache_dir
        )

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        """
        Save the extractor prompts to a path.
        """
        self.extractor_prompt.save(cache_dir)
