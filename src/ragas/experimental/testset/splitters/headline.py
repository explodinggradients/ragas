import re
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ragas.experimental.testset.graph import Node, Relationship


@dataclass
class Splitter(ABC):
    @abstractmethod
    def split(self, node: Node) -> t.Tuple[t.List[Node], t.List[Relationship]]:
        pass


@dataclass
class HeadlineSplitter(Splitter):
    def split(self, node: Node) -> t.Tuple[t.List[Node], t.List[Relationship]]:
        text = node.get_property("page_content")
        return self._split_text_by_headlines(text, node.get_property("headlines"))

    def _find_headline_indices(self, text: str, headlines: t.List[str]) -> t.Dict:
        index_dict = {}
        for headline in headlines:
            # Build a regex pattern to match the headline with newlines before and after
            pattern = rf"(?<=\n){re.escape(headline)}"
            matches = re.finditer(pattern, text)
            first_match = next(matches, None)
            if first_match:
                index_dict[headline] = first_match.start()
        return index_dict

    def _split_text_by_headlines(
        self, text: str, headlines: t.List[str]
    ) -> t.List[t.Tuple[str, str]]:
        indices = []
        seperators = []
        headline_indices = self._find_headline_indices(text, headlines)
        values = list(headline_indices.values())
        keys = list(headline_indices.keys())
        values.append(len(text))
        for key, start_idx, end_idx in zip(keys, values[:-1], values[1:]):
            indices.append((start_idx, end_idx))
            seperators.append(key)

        chunks = [(text[idx[0] : idx[1]], sep) for idx, sep in zip(indices, seperators)]
        return chunks
