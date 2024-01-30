from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ragas.llms.json_load import json_loader
from ragas.testset.prompts import keyphrase_extraction_prompt

if t.TYPE_CHECKING:
    from ragas.llms.base import BaseRagasLLM
    from ragas.testset.docstore import Node


@dataclass
class Extractor(ABC):
    llm: BaseRagasLLM

    @abstractmethod
    def extract(self, node: Node) -> t.Any:
        ...


class keyphraseExtractor(Extractor):
    async def extract(self, node: Node) -> t.List[str]:
        prompt = keyphrase_extraction_prompt.format(text=node.page_content)
        results = await self.llm.agenerate_text(prompt=prompt)
        keyphrases = json_loader.sync_safe_load(
            results.generations[0][0].text.strip(), llm=self.llm
        )
        return keyphrases.get("keyphrases", [])
