import re
import typing as t
from collections import defaultdict
from dataclasses import dataclass

from ragas.experimental.testset.extractors.base import BaseExtractor
from ragas.experimental.testset.graph import Node


@dataclass
class RegexBasedExtractor(BaseExtractor):
    pattern: str = ""
    is_multiline: bool = False

    async def _extract(self, node: Node) -> t.Tuple[str, t.Any]:
        text = node.get_property("page_content")

        matches = (
            re.finditer(self.pattern, text, re.MULTILINE)
            if self.is_multiline
            else re.finditer(self.pattern, text)
        )
        result = defaultdict(list)
        for m in matches:
            m_dict = {k: v for k, v in m.groupdict().items() if v is not None}
            for key, value in m_dict.items():
                result[key].append(value)
        return result
