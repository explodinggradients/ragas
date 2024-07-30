import re
import typing as t
from collections import defaultdict
from dataclasses import dataclass

from langchain_core.documents import Document as LCDocument
from ragas_experimental.testset.extractors.base import Extractor, Regex
from ragas_experimental.testset.graph import Node


@dataclass
class RulebasedExtractor(Extractor):
    regex: t.Optional[Regex] = None
    is_multiline: bool = False

    def __post_init__(self):
        assert self.regex is not None, "Regex pattern is not initialized"
        self.pattern = self.regex()

    async def aextract_text(self, text):
        raise NotImplementedError(
            "aextract() is not implemented for RulebasedExtractor"
        )

    def extract_text(self, text):
        matches = (
            re.finditer(self.pattern, text, re.MULTILINE)
            if self.is_multiline
            else re.finditer(self.pattern, text)
        )
        result = defaultdict(list)
        for m in matches:
            m = {k: v for k, v in m.groupdict().items() if v is not None}
            for key in m:
                result[key].append(m[key])

        return result

    def extract(self, node: t.Union[Node, LCDocument]) -> t.Any:
        return super().extract(node)

    def merge_extractors(self, *extractors) -> t.List[Extractor]:
        if isinstance(
            self, RulebasedExtractor
        ):  # Check if called by an initiated class
            extractors = (self,) + extractors

        assert all(
            isinstance(extractor, RulebasedExtractor) for extractor in extractors
        ), "All extractors must be of type RulebasedExtractor"

        final_extractors: t.List[t.List[RulebasedExtractor]] = []
        added_indices = []
        for idx, extractor in enumerate(extractors):
            if idx not in added_indices:
                final_extractors.append([extractor])
                added_indices.append(idx)
                other_extractors = [
                    ext for i, ext in enumerate(extractors) if i not in added_indices
                ]
                filtered_extractors = [
                    ext
                    for ext in other_extractors
                    if extractor.attribute == ext.attribute
                    if extractor.is_multiline == ext.is_multiline
                ]
                for ext in filtered_extractors:
                    final_extractors[-1].append(ext)
                    added_indices.append(extractors.index(ext))

        extractors_to_return = []
        for extractors in final_extractors:
            if len(extractors) > 1:
                pattern = "|".join([extractor.pattern for extractor in extractors])
                updated_regex = Regex(name="merged_extractor", pattern=pattern)
            else:
                pattern = extractors[0].pattern
                updated_regex = extractors[0].regex
            extractors_to_return.append(
                RulebasedExtractor(
                    attribute=extractors[0].attribute,
                    regex=updated_regex,
                    is_multiline=extractors[0].is_multiline,
                )
            )
        return extractors_to_return


links_extractor_pattern = r"(?i)\b(?:https?://|www\.)\S+\b"
emails_extractor_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
markdown_headings = r"^(#{1,6})\s+(.*)"

email_extractor = RulebasedExtractor(
    regex=Regex(name="email", pattern=emails_extractor_pattern)
)
link_extractor = RulebasedExtractor(
    regex=Regex(name="link", pattern=links_extractor_pattern)
)
markdown_headings = RulebasedExtractor(
    regex=Regex(name="markdown_headings", pattern=markdown_headings), is_multiline=True
)
