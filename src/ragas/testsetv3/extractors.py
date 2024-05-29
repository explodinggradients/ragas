import re
import typing as t
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from langchain_core.documents import Document as LCDocument

from ragas.llms.base import BaseRagasLLM, llm_factory
from ragas.llms.json_load import json_loader
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
        return f"(?P<{self.name}>{self.pattern})" if self.name != "merged_extractor" else self.pattern



class Extractor(ABC):
    @abstractmethod
    def extract(self, text) -> t.Any:
        pass

    @abstractmethod
    def merge_extractors(self, *extractors):
        pass
    
@dataclass
class RulebasedExtractor(Extractor):
    regex: Regex

    def extract(self, text):
        matches = re.finditer(self.regex(), text)
        result = defaultdict(list)
        for m in matches:
            m = {k:v for k, v in m.groupdict().items() if v is not None}
            for key in m:
                result[key].append(m[key])
            
        return result

    def merge_extractors(self, *extractors):  # Instance-level method
        if isinstance(self, RulebasedExtractor):  # Check if called by an initiated class
            extractors = (self,) + extractors
        pattern = "|".join([extractor.regex() for extractor in extractors])
        updated_regex = Regex(name="merged_extractor", pattern=pattern)
        return RulebasedExtractor(regex=updated_regex)



@dataclass
class LLMbasedExtractor(Extractor):
    prompt: Prompt
    llm: t.Optional[BaseRagasLLM] = None

    async def extract(self, text, is_asycn=True):
        if self.llm is None:
            self.llm = llm_factory()

        output = await self.llm.generate(
            prompt=self.prompt.format(text=text), is_async=is_asycn
        )
        output = output.generations[0][0].text.strip()
        if self.prompt.output_type == "json":
            return await json_loader.safe_load(
                output, self.llm
            )
        else:
            return {self.prompt.name: output}

    def merge_extractors(self, *extractors):
        
        if isinstance(self, LLMbasedExtractor):
            extractors = (self,) + extractors
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

        return LLMbasedExtractor(prompt=prompt)


@dataclass
class DocumentExtractor():
    extractors: t.List[Extractor]
    
    def __post_init__(self):
        llm_extractor = [extractor for extractor in self.extractors if isinstance(extractor, LLMbasedExtractor)]
        rule_extractor = [extractor for extractor in self.extractors if isinstance(extractor, RulebasedExtractor)]
        self.llm_extractors = LLMbasedExtractor.merge_extractors(*llm_extractor) if llm_extractor else None
        self.regex_extractors = RulebasedExtractor.merge_extractors(*rule_extractor) if rule_extractor else None
        
    async def __call__(self, documents: t.Sequence[LCDocument]):
        
        for doc in documents:
            if self.llm_extractors:
                output = await self.llm_extractors.extract(doc.page_content)
                doc.metadata.update(output)
            if self.regex_extractors:
                output = self.regex_extractors.extract(doc.page_content)
                doc.metadata.update(output)
            
        return documents
    
            
summary_extractor = LLMbasedExtractor(prompt=summary_extactor_prompt)
headline_extractor = LLMbasedExtractor(prompt=headline_extractor_prompt)

email_extractor = RulebasedExtractor(
    Regex(name="email", pattern=emails_extractor_pattern)
)
link_extractor = RulebasedExtractor(Regex(name="link", pattern=links_extractor_pattern))


if __name__ == "__main__":
    doc_extractor = DocumentExtractor(extractors=[summary_extractor, link_extractor, headline_extractor])