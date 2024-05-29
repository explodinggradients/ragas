import re
import typing as t
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass

from ragas.llms.base import BaseRagasLLM, llm_factory
from ragas.llms.prompt import Prompt

RULE_BASED_EXTRACTORS = [
    "email_extractor",
    "link_extractor",
]

LLM_EXTRACTORS = [
    "summary_extractor",
    "entity_extractor",
    "keyphrase_extractor",
    "headline_extractor",
]


summary_extactor_prompt = Prompt(
    name="summary_extractor",
    instruction="Summarize the given text in less than 10 sentences.",
    examples=[
        {
            "text": "The quick brown fox jumps over the lazy dog.",
            "summary": "The quick brown fox jumps over the lazy dog.",
        }
    ],
    input_keys=["text"],
    output_key="summary",
    output_type="str",
)

headline_extractor_prompt = Prompt(
    name="headline_extractor",
    instruction="Extract H1 headlines from the given text.",
    examples=[
        {
            "text": "The quick brown fox jumps over the lazy dog.",
            "headlines": ["headline1", "headline2"],
        }
    ],
    input_keys=["text"],
    output_key="headlines",
    output_type="json",
)

links_extractor_pattern = r"(?i)\b(?:https?://|www\.)\S+\b"
emails_extractor_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"


@dataclass
class Regex:
    name: str
    pattern: str

    def __call__(self):
        # Ensure the pattern is a raw string
        if not isinstance(self.pattern, str):
            raise TypeError("Pattern must be a string.")

        if not isinstance(self.name, str):
            raise TypeError("Group name must be a string.")

        # Add the named group to the pattern
        return f"(?P<{self.name}>{self.pattern})"


class Extractor(ABC):
    @abstractmethod
    def extract(self, text) -> t.Dict[t.Any, t.Any]:
        pass

    @classmethod
    def merge_extractors(cls, *extractors):
        pass


@dataclass
class RulebasedExtractor(Extractor):
    regex: Regex

    def extract(self, text):
        matches = re.finditer(self.regex(), text)
        result = defaultdict(list)
        for match in matches:
            key, val = match.groupdict()
            result[key].append(val)

        return result

    @classmethod
    def merge_extractor(cls, *extractors):
        pattern = "|".join([extractor.regex() for extractor in extractors])
        updated_regex = Regex(name="merged_extractor", pattern=pattern)
        return cls(regex=updated_regex)


@dataclass
class LLMbasedExtractor(Extractor):
    prompt: Prompt
    llm: t.Optional[BaseRagasLLM] = None

    def extract(self, text, is_asycn=True):
        if self.llm is None:
            self.llm = llm_factory()

        output = self._extract_async(text, is_asycn=is_asycn)

        return output.generations[0][0].text.strip()

    async def _extract_async(self, text, is_asycn):
        assert self.llm is not None, "LLM model is not initialized."

        return await self.llm.generate(
            prompt=self.prompt.format(text=text), is_async=is_asycn
        )

    @classmethod
    def merge_extractors(cls, *extractors):
        if not any(hasattr(extractor, "prompt") for extractor in extractors):
            raise ValueError("Both extractors should have a prompt attribute.")

        if len({tuple(extractor.prompt.input_keys) for extractor in extractors}) != 1:
            raise ValueError("All extractors should have the same input keys.")

        if len({len(extractor.prompt.examples) for extractor in extractors}) != 1:
            raise ValueError("All extractors should have the same number of examples.")

        instruction = "\n".join(
            [
                f"{i}:{extractor.prompt.instruction}"
                for i, extractor in enumerate(extractors)
            ]
        )

        examples = []
        for idx, example in enumerate(extractors[0].prompt.examples):
            example = {key: example[key] for key in extractors[0].prompt.input_keys}
            output = {
                extractor.prompt.output_key: extractor.prompt.examples[idx][
                    extractor.prompt.output_key
                ]
                for extractor in extractors
            }
            example.update({"output": output})
            examples.append(example)

        prompt = Prompt(
            name="merged_extractor",
            instruction=instruction,
            examples=examples,
            input_keys=extractors[0].prompt.input_keys,
            output_key="output",
            output_type="json",
        )

        return cls(prompt=prompt)


summary_extractor = LLMbasedExtractor(prompt=summary_extactor_prompt)
headline_extractor = LLMbasedExtractor(prompt=headline_extractor_prompt)
merged_extractor = LLMbasedExtractor.merge_extractors(
    summary_extractor, headline_extractor
)

email_extractor = RulebasedExtractor(
    Regex(name="email", pattern=emails_extractor_pattern)
)
link_extractor = RulebasedExtractor(Regex(name="link", pattern=links_extractor_pattern))
merged_rule_extractor = RulebasedExtractor.merge_extractor(
    email_extractor, link_extractor
)
