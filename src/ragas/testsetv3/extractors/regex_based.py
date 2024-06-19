import re
from collections import defaultdict
from dataclasses import dataclass

from ragas.testsetv3.extractors.base import Extractor, Regex


@dataclass
class RulebasedExtractor(Extractor):
    regex: Regex

    def extract(self, text):
        matches = re.finditer(self.regex(), text)
        result = defaultdict(list)
        for m in matches:
            m = {k: v for k, v in m.groupdict().items() if v is not None}
            for key in m:
                result[key].append(m[key])

        return result

    def merge_extractors(self, *extractors):
        if isinstance(
            self, RulebasedExtractor
        ):  # Check if called by an initiated class
            extractors = (self,) + extractors
        pattern = "|".join([extractor.regex() for extractor in extractors])
        updated_regex = Regex(name="merged_extractor", pattern=pattern)
        return [RulebasedExtractor(regex=updated_regex)]


links_extractor_pattern = r"(?i)\b(?:https?://|www\.)\S+\b"
emails_extractor_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"

email_extractor = RulebasedExtractor(
    Regex(name="email", pattern=emails_extractor_pattern)
)
link_extractor = RulebasedExtractor(Regex(name="link", pattern=links_extractor_pattern))
