import typing as t
from dataclasses import dataclass

import numpy as np
import tiktoken
from langchain_core.documents import Document as LCDocument
from ragas_experimental.testset.extractors.base import Extractor
from ragas_experimental.testset.extractors.prompts import (
    headline_extractor_prompt,
    keyphrase_extractor_prompt,
    summary_extactor_prompt,
    title_extractor_prompt,
)
from ragas_experimental.testset.graph import Node
from ragas_experimental.testset.utils import MODEL_MAX_LENGTHS, merge_dicts

from ragas.llms.base import BaseRagasLLM, llm_factory
from ragas.llms.json_load import json_loader
from ragas.llms.prompt import Prompt


@dataclass
class LLMbasedExtractor(Extractor):
    prompt: t.Optional[Prompt] = None
    llm: t.Optional[BaseRagasLLM] = None

    def __post_init__(self):
        if self.llm is None:
            self.llm = llm_factory()
        assert self.prompt is not None, "Prompt is not initialized"

    def extract_text(self, text: str) -> t.Any:
        raise NotImplementedError("extract() is not implemented for LLMbasedExtractor")

    async def aextract(self, node: t.Union[Node, LCDocument]) -> t.Any:
        return await super().aextract(node)

    async def _generate_output(self, p_value, is_asycn=True):
        assert self.llm is not None, "LLM is not initialized"
        assert self.prompt is not None, "Prompt must not be None"

        output = await self.llm.generate(prompt=p_value, is_async=is_asycn)
        output = output.generations[0][0].text.strip()
        if self.prompt.output_type == "json":
            return await json_loader.safe_load(output, self.llm)

        return {self.prompt.name: output}

    async def aextract_text(self, text):
        is_asycn = True
        if self.llm is None:
            self.llm = llm_factory()

        assert self.prompt is not None, "Prompt is not initialized"

        # TODO: handle different models
        model_name = "gpt-3.5-turbo-"
        model_max_length = MODEL_MAX_LENGTHS.get(model_name, 8000)
        model_input_length = model_max_length - (model_max_length // 4)

        enc = tiktoken.encoding_for_model(model_name)
        p_value = self.prompt.format(text=text)
        tokens = enc.encode(p_value.to_string())
        prompt_length = len(tokens)
        ratio = prompt_length / model_input_length

        # TODO modify to suit abstractive tasks as well
        if ratio > 1:
            max_tokens_per_run = int(np.ceil(prompt_length / np.ceil(ratio)))
            inputs = [
                enc.decode(tokens[i : i + max_tokens_per_run])
                for i in range(0, len(tokens), max_tokens_per_run)
            ]
            inputs = [self.prompt.format(text=inp) for inp in inputs]
            outputs = [await self._generate_output(inp, is_asycn) for inp in inputs]
            output = merge_dicts(*outputs)

        else:
            output = await self._generate_output(p_value, is_asycn)

        return output

    def merge_extractors(self, *extractors):
        if isinstance(self, LLMbasedExtractor):
            extractors = (self,) + extractors

        final_extractors: t.List[t.List[LLMbasedExtractor]] = []
        added_indices = []

        extractors = list(extractors)
        for idx, extractor in enumerate(extractors):
            if idx not in added_indices:
                final_extractors.append([extractor])
                added_indices.append(idx)
                other_extractors = [
                    ext for i, ext in enumerate(extractors) if i not in added_indices
                ]

                assert extractor.prompt is not None, "Input keys are not defined"

                input_keys = extractor.prompt.input_keys
                filtered_extractors = [
                    ext
                    for ext in other_extractors
                    if ext.prompt
                    and ext.prompt.input_keys == input_keys
                    and len(ext.prompt.examples) == len(extractor.prompt.examples)
                    and extractor.attribute == ext.attribute
                ]
                for ext in filtered_extractors:
                    assert ext.prompt is not None, "Prompt is not defined for extractor"
                    input_values = [
                        ext.prompt.examples[i][key]
                        for i in range(len(ext.prompt.examples))
                        for key in input_keys
                    ]
                    if all(
                        extractor.prompt.examples[i][key] == input_values[i]
                        for i in range(len(ext.prompt.examples))
                        for key in input_keys
                    ):
                        final_extractors[-1].append(ext)
                        added_indices.append(extractors.index(ext))

        extractors_to_return = []
        for extractors in final_extractors:
            instruction = "\n".join(
                [
                    f"{i}:{extractor.prompt.instruction}"
                    for i, extractor in enumerate(extractors)
                    if extractor.prompt
                ]
            )

            examples = []
            extractor_prompt1 = extractors[0].prompt if extractors[0].prompt else None
            if extractor_prompt1 is not None:
                for idx, example in enumerate(extractor_prompt1.examples):
                    example = {
                        key: example[key] for key in extractor_prompt1.input_keys
                    }
                    output = {
                        extractor.prompt.output_key: extractor.prompt.examples[idx][
                            extractor.prompt.output_key
                        ]
                        for extractor in extractors
                        if extractor.prompt
                    }
                    example.update({"output": output})
                    examples.append(example)

                prompt = Prompt(
                    name="merged_extractor",
                    instruction=instruction,
                    examples=examples,
                    input_keys=extractor_prompt1.input_keys,
                    output_key="output",
                    output_type="json",
                )
                extractors_to_return.append(
                    LLMbasedExtractor(attribute=extractors[0].attribute, prompt=prompt)
                )

        return extractors_to_return


summary_extractor = LLMbasedExtractor(prompt=summary_extactor_prompt)
headline_extractor = LLMbasedExtractor(prompt=headline_extractor_prompt)
keyphrase_extractor = LLMbasedExtractor(prompt=keyphrase_extractor_prompt)
title_extractor = LLMbasedExtractor(prompt=title_extractor_prompt)
