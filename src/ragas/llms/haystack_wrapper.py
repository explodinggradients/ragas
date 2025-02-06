import typing as t

try:
    from haystack_experimental.core import AsyncPipeline
except ImportError:
    raise ImportError(
        "haystack-experimental is not installed. Please install it using `pip install haystack-experimental==0.4.0`."
    )
try:
    from haystack.components.generators import (  # type: ignore
        AzureOpenAIGenerator,
        HuggingFaceAPIGenerator,
        HuggingFaceLocalGenerator,
        OpenAIGenerator,
    )
except ImportError:
    raise ImportError(
        "pip install haystack-ai is not installed. Please install it using `pip install pip install haystack-ai`."
    )
from langchain_core.callbacks import Callbacks
from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import PromptValue

from ragas.cache import CacheInterface
from ragas.llms import BaseRagasLLM
from ragas.run_config import RunConfig


class HaystackLLMWrapper(BaseRagasLLM):
    def __init__(
        self,
        haystack_generator: t.Union[
            OpenAIGenerator,  # type: ignore
            AzureOpenAIGenerator,  # type: ignore
            HuggingFaceAPIGenerator,  # type: ignore
            HuggingFaceLocalGenerator,  # type: ignore
        ],
        run_config: t.Optional[RunConfig] = None,
        cache: t.Optional[CacheInterface] = None,
    ):
        super().__init__(cache=cache)

        self.haystack_client = haystack_generator

        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)
        self.generator = haystack_generator
        self.async_pipeline = AsyncPipeline()
        self.async_pipeline.add_component("llm", self.generator)

    def is_finished(self, response: LLMResult) -> bool:
        return True

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ):
        hs_response = self.haystack_client.run(prompt.to_string())
        return LLMResult(generations=[[Generation(text=hs_response)]])

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ):
        async def llm_pipeline(query: str):
            result = ""
            async for output in self.async_pipeline.run({"llm": {"prompt": query}}):
                result = output["llm"]["replies"][0]
            return result

        hs_response = await llm_pipeline(query=prompt.to_string())
        return LLMResult(generations=[[Generation(text=hs_response)]])

    def set_run_config(self, run_config: RunConfig):
        self.run_config = run_config
        pass

    def __repr__(self):
        if isinstance(self.generator, (OpenAIGenerator, HuggingFaceLocalGenerator)):  # type: ignore
            model = self.generator.model
        elif isinstance(self.generator, HuggingFaceAPIGenerator):  # type: ignore
            model = self.generator.api_params
        elif isinstance(self.generator, AzureOpenAIGenerator):  # type: ignore
            model = self.generator.azure_deployment
        else:
            model = "Unknown"

        return f"{self.__class__}(llm={model}(...))"
