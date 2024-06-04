import re
import typing as t
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
import numpy as np

from langchain_core.documents import Document as LCDocument

from ragas.llms.base import BaseRagasLLM, llm_factory
from ragas.llms.json_load import json_loader
from ragas.llms.prompt import Prompt
from ragas.testsetv3.utils import merge_dicts, MODEL_MAX_LENGTHS

import tiktoken
from ragas.embeddings.base import embedding_factory, BaseRagasEmbeddings

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
            "text": "Artificial intelligence\n\nArtificial intelligence is transforming various industries by automating tasks that previously required human intelligence. From healthcare to finance, AI is being used to analyze vast amounts of data quickly and accurately. This technology is also driving innovations in areas like self-driving cars and personalized recommendations.",
            "summary": "AI is revolutionizing industries by automating tasks, analyzing data, and driving innovations like self-driving cars and personalized recommendations.",
        }
    ],
    input_keys=["text"],
    output_key="summary",
    output_type="str",
)

headline_extractor_prompt = Prompt(
    name="headline_extractor",
    instruction="Extract H2 headlines from the given text.",
    examples=[
        {
            "text": "Artificial intelligence\n\nArtificial intelligence is transforming various industries by automating tasks that previously required human intelligence. From healthcare to finance, AI is being used to analyze vast amounts of data quickly and accurately. This technology is also driving innovations in areas like self-driving cars and personalized recommendations.",
            "headlines": ["Artificial intelligence"],
        }
    ],
    input_keys=["text"],
    output_key="headlines",
    output_type="json",
)

keyphrase_extractor_prompt = Prompt(
    name="keyphrase_extractor",
    instruction="Extract keyphrases from the given text.",
    examples=[
        {
            "text": "Artificial intelligence\n\nArtificial intelligence is transforming various industries by automating tasks that previously required human intelligence. From healthcare to finance, AI is being used to analyze vast amounts of data quickly and accurately. This technology is also driving innovations in areas like self-driving cars and personalized recommendations.",
            "keyphrases": ["Artificial intelligence", "automating tasks", "healthcare", "finance", "analyze data", "self-driving cars", "personalized recommendations"],
        }
    ],
    input_keys=["text"],
    output_key="keyphrases",
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
        return (
            f"(?P<{self.name}>{self.pattern})"
            if self.name != "merged_extractor"
            else self.pattern
        )


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
            m = {k: v for k, v in m.groupdict().items() if v is not None}
            for key in m:
                result[key].append(m[key])

        return result

    def merge_extractors(self, *extractors):  # Instance-level method
        if isinstance(
            self, RulebasedExtractor
        ):  # Check if called by an initiated class
            extractors = (self,) + extractors
        pattern = "|".join([extractor.regex() for extractor in extractors])
        updated_regex = Regex(name="merged_extractor", pattern=pattern)
        return RulebasedExtractor(regex=updated_regex)


@dataclass
class LLMbasedExtractor(Extractor):
    prompt: Prompt
    llm: t.Optional[BaseRagasLLM] = None
    
    
    async def _generate_output(self, p_value, is_asycn=True):
        assert self.llm is not None, "LLM is not initialized"
        
        output = await self.llm.generate(
                prompt=p_value, is_async=is_asycn
            )
        output = output.generations[0][0].text.strip()
        if self.prompt.output_type == "json":
            return await json_loader.safe_load(output, self.llm)
        
        return {self.prompt.name: output}
    

    async def extract(self, text, is_asycn=True):
        if self.llm is None:
            self.llm = llm_factory()
            
        model_name = self.llm.langchain_llm.model_name or "gpt-2"
        model_max_length = MODEL_MAX_LENGTHS.get(model_name, 8000)
        model_input_length = model_max_length - (model_max_length//4)
        
        enc = tiktoken.encoding_for_model(model_name)
        p_value = self.prompt.format(text=text)
        tokens = enc.encode(p_value.to_string())
        prompt_length = len(tokens) 
        ratio = prompt_length / model_input_length
        
        #TODO modify to suit abstractive tasks as well
        if ratio > 1:
            max_tokens_per_run = int(np.ceil(prompt_length / np.ceil(ratio)))
            inputs = [enc.decode(tokens[i:i+max_tokens_per_run]) for i in range(0, len(tokens), max_tokens_per_run)]
            inputs = [self.prompt.format(text=inp) for inp in inputs]
            outputs = [await self._generate_output(inp, is_asycn) for inp in inputs]
            output = merge_dicts(*outputs)
            
        else:
            output = await self._generate_output(p_value, is_asycn)
            
        return output

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
class DocumentExtractor:
    extractors: t.List[Extractor]
    embedding: t.Optional[BaseRagasEmbeddings] = None

    def __post_init__(self):
        llm_extractor = [
            extractor
            for extractor in self.extractors
            if isinstance(extractor, LLMbasedExtractor)
        ]
        rule_extractor = [
            extractor
            for extractor in self.extractors
            if isinstance(extractor, RulebasedExtractor)
        ]
        self.llm_extractors = (
            LLMbasedExtractor.merge_extractors(*llm_extractor)
            if llm_extractor
            else None
        )
        self.regex_extractors = (
            RulebasedExtractor.merge_extractors(*rule_extractor)
            if rule_extractor
            else None
        )

    async def extract(self, documents: t.Sequence[LCDocument]):
        for doc in documents:
            if self.llm_extractors:
                output = await self.llm_extractors.extract(doc.page_content)
                doc.metadata.update(output)
            if self.regex_extractors:
                output = self.regex_extractors.extract(doc.page_content)
                doc.metadata.update(output)

        doc = documents[0]
        extractive_metadata_keys = []
        for metadata in doc.metadata:
            if isinstance(doc.metadata[metadata], str):
                idx = doc.page_content.find(doc.metadata[metadata])
                if idx != -1:
                    extractive_metadata_keys.append(metadata)
            elif isinstance(doc.metadata[metadata], list):
                idx = [doc.page_content.find(item) for item in doc.metadata[metadata]]
                if sum(i != -1 for i in idx) > len(idx) / 2:
                    extractive_metadata_keys.append(metadata)
                    
        for doc in documents:
            doc.metadata["extractive_metadata_keys"] = extractive_metadata_keys

        return documents
    
    async def embed(self, documents: t.Sequence[LCDocument], attributes=t.List[str]):
        
        self.embedding = self.embedding if self.embedding is not None else embedding_factory()
        for attr in attributes:
            if attr == "page_content":
                items_to_embed = [doc.page_content for doc in documents]
            else:
                items_to_embed = [doc.metadata.get(attr,"") for doc in documents]
                
            embeddings_list = await self.embedding.aembed_documents(items_to_embed)
            assert len(embeddings_list) == len(items_to_embed), "Embeddings and document must be of equal length"
            for doc, embedding in zip(documents, embeddings_list):
                doc.metadata[f"{attr}_embedding"] = embedding
            
        return documents


summary_extractor = LLMbasedExtractor(prompt=summary_extactor_prompt)
headline_extractor = LLMbasedExtractor(prompt=headline_extractor_prompt)
keyphrase_extractor = LLMbasedExtractor(prompt=keyphrase_extractor_prompt)

email_extractor = RulebasedExtractor(
    Regex(name="email", pattern=emails_extractor_pattern)
)
link_extractor = RulebasedExtractor(Regex(name="link", pattern=links_extractor_pattern))


if __name__ == "__main__":
    doc_extractor = DocumentExtractor(
        extractors=[summary_extractor, link_extractor, headline_extractor]
    )
