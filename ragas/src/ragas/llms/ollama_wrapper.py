import typing as t
from langchain_core.callbacks import Callbacks
from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import PromptValue
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from ragas.cache import CacheInterface
from ragas.llms import BaseRagasLLM
from ragas.run_config import RunConfig


class OllamaLLMWrapper(BaseRagasLLM):
    """
    A wrapper class for using Ollama LLM within the Ragas framework.

    This class integrates Ollama's LLM into Ragas, enabling both synchronous and
    asynchronous text generation.

    Parameters
    ----------
    ollama_llm : ChatOllama
        An instance of Ollama chat model.
    run_config : RunConfig, optional
        Configuration object to manage LLM execution settings, by default None.
    cache : CacheInterface, optional
        A cache instance for storing results, by default None.
    """

    def __init__(
        self,
        ollama_llm: ChatOllama,
        run_config: t.Optional[RunConfig] = None,
        cache: t.Optional[CacheInterface] = None,
    ):
        super().__init__(cache=cache)
        self.llm = ollama_llm  # 直接用 ChatOllama 实例
        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)

    def is_finished(self, response: LLMResult) -> bool:
        """Check if the generation is finished."""
        return True

    def generate_text(
        self,
        prompt: str,
        stop: t.Optional[t.List[str]] = None,
        run_manager: t.Optional[Callbacks] = None,
        **kwargs: t.Any,
    ) -> LLMResult:
        """Generate text from the model."""
        response = self.llm.invoke(prompt)
        print("LLM raw output:", response.content)
        return LLMResult(generations=[[Generation(text=response.content)]])

    async def agenerate_text(
        self,
        prompt: str,
        stop: t.Optional[t.List[str]] = None,
        run_manager: t.Optional[Callbacks] = None,
        **kwargs: t.Any,
    ) -> LLMResult:
        """Generate text from the model asynchronously."""
        response = await self.llm.ainvoke(prompt)
        print("LLM raw output:", response.content)
        return LLMResult(generations=[[Generation(text=response.content)]])

    def _generate(
        self,
        prompts: t.List[PromptValue],
        stop: t.Optional[t.List[str]] = None,
        run_manager: t.Optional[Callbacks] = None,
        **kwargs: t.Any,
    ) -> LLMResult:
        """Generate text from the model."""
        generations = []
        for prompt in prompts:
            response = self.llm.invoke(prompt.to_messages())
            generations.append([Generation(text=response.content)])
        return LLMResult(generations=generations)

    async def _agenerate(
        self,
        prompts: t.List[PromptValue],
        stop: t.Optional[t.List[str]] = None,
        run_manager: t.Optional[Callbacks] = None,
        **kwargs: t.Any,
    ) -> LLMResult:
        """Generate text from the model asynchronously."""
        generations = []
        for prompt in prompts:
            response = await self.llm.ainvoke(prompt.to_messages())
            generations.append([Generation(text=response.content)])
        return LLMResult(generations=generations)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(llm={self.llm.model}(...))" 