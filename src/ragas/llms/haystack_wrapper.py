import typing as t

from langchain_core.callbacks import Callbacks
from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import PromptValue

from ragas.cache import CacheInterface
from ragas.llms import BaseRagasLLM
from ragas.run_config import RunConfig


def _extract_haystack_reply_text(reply: t.Any) -> str:
    if reply is None:
        return ""
    if isinstance(reply, str):
        return reply
    text = getattr(reply, "text", None)
    if isinstance(text, str):
        return text
    return str(reply)


def _get_haystack_generator_types() -> tuple[tuple[type, ...], tuple[type, ...]]:
    from haystack.components.generators.azure import AzureOpenAIGenerator
    from haystack.components.generators.chat.azure import AzureOpenAIChatGenerator
    from haystack.components.generators.chat.azure_responses import (
        AzureOpenAIResponsesChatGenerator,
    )
    from haystack.components.generators.chat.fallback import FallbackChatGenerator
    from haystack.components.generators.chat.hugging_face_api import (
        HuggingFaceAPIChatGenerator,
    )
    from haystack.components.generators.chat.hugging_face_local import (
        HuggingFaceLocalChatGenerator,
    )
    from haystack.components.generators.chat.openai import OpenAIChatGenerator
    from haystack.components.generators.chat.openai_responses import (
        OpenAIResponsesChatGenerator,
    )
    from haystack.components.generators.hugging_face_api import (
        HuggingFaceAPIGenerator,
    )
    from haystack.components.generators.hugging_face_local import (
        HuggingFaceLocalGenerator,
    )
    from haystack.components.generators.openai import OpenAIGenerator

    legacy_generators = (
        AzureOpenAIGenerator,
        HuggingFaceAPIGenerator,
        HuggingFaceLocalGenerator,
        OpenAIGenerator,
    )
    chat_generators = (
        AzureOpenAIChatGenerator,
        AzureOpenAIResponsesChatGenerator,
        FallbackChatGenerator,
        HuggingFaceAPIChatGenerator,
        HuggingFaceLocalChatGenerator,
        OpenAIChatGenerator,
        OpenAIResponsesChatGenerator,
    )
    return legacy_generators, chat_generators


def _haystack_generator_model_name(generator: t.Any) -> str:
    from haystack.components.generators.azure import AzureOpenAIGenerator
    from haystack.components.generators.hugging_face_api import (
        HuggingFaceAPIGenerator,
    )
    from haystack.components.generators.hugging_face_local import (
        HuggingFaceLocalGenerator,
    )
    from haystack.components.generators.openai import OpenAIGenerator

    if isinstance(generator, OpenAIGenerator):
        return str(generator.model)
    if isinstance(generator, HuggingFaceLocalGenerator):
        return str(generator.huggingface_pipeline_kwargs.get("model", "Unknown"))
    if isinstance(generator, HuggingFaceAPIGenerator):
        return str(generator.api_params.get("model", "Unknown"))
    if isinstance(generator, AzureOpenAIGenerator):
        return str(generator.azure_deployment)

    model = getattr(generator, "model", None)
    if isinstance(model, str):
        return model
    azure_deployment = getattr(generator, "azure_deployment", None)
    if isinstance(azure_deployment, str):
        return azure_deployment
    return "Unknown"


class HaystackLLMWrapper(BaseRagasLLM):
    """
    A wrapper class for using Haystack LLM generators within the Ragas framework.

    This class integrates Haystack's legacy and chat generator components into
    Ragas, enabling both synchronous and asynchronous text generation.

    Parameters
    ----------
    haystack_generator : Haystack generator instance
        A Haystack legacy generator (for example ``OpenAIGenerator``) or chat
        generator (for example ``OpenAIChatGenerator``).
    run_config : RunConfig, optional
        Configuration object to manage LLM execution settings, by default None.
    cache : CacheInterface, optional
        A cache instance for storing results, by default None.
    """

    def __init__(
        self,
        haystack_generator: t.Any,
        run_config: t.Optional[RunConfig] = None,
        cache: t.Optional[CacheInterface] = None,
    ):
        super().__init__(cache=cache)

        try:
            from haystack import AsyncPipeline
        except ImportError as exc:
            raise ImportError(
                "Haystack is not installed. Please install it using `pip install haystack-ai`."
            ) from exc

        legacy_generators, chat_generators = _get_haystack_generator_types()
        supported_generators = legacy_generators + chat_generators

        if not isinstance(haystack_generator, supported_generators):
            supported_names = ", ".join(
                generator_type.__name__ for generator_type in supported_generators
            )
            raise TypeError(
                "Expected 'haystack_generator' to be a supported Haystack generator, "
                f"but received {type(haystack_generator).__name__}. "
                f"Supported types: {supported_names}."
            )

        self._uses_chat_generator = isinstance(haystack_generator, chat_generators)
        self.generator = haystack_generator
        self.async_pipeline = AsyncPipeline()
        self.async_pipeline.add_component("llm", self.generator)  # type: ignore[reportArgumentType]

        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)

    def is_finished(self, response: LLMResult) -> bool:
        return True

    def _pipeline_llm_input(
        self, prompt: PromptValue, temperature: t.Optional[float]
    ) -> dict[str, t.Any]:
        prompt_text = prompt.to_string()
        llm_input: dict[str, t.Any] = {}
        if self._uses_chat_generator:
            llm_input["messages"] = prompt_text
        else:
            llm_input["prompt"] = prompt_text
        if temperature is not None:
            llm_input["generation_kwargs"] = {"temperature": temperature}
        return llm_input

    def _replies_from_output(self, component_output: dict[str, t.Any]) -> list[t.Any]:
        replies = component_output.get("replies", [])
        if not replies:
            replies = component_output.get("llm", {}).get("replies", [])
        return t.cast(list[t.Any], replies)

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks: t.Optional[Callbacks] = None,
    ) -> LLMResult:
        run_kwargs: dict[str, t.Any] = {}
        if self._uses_chat_generator:
            run_kwargs["messages"] = prompt.to_string()
        else:
            run_kwargs["prompt"] = prompt.to_string()
        if temperature is not None:
            run_kwargs["generation_kwargs"] = {"temperature": temperature}

        component_output = self.generator.run(**run_kwargs)
        replies = self._replies_from_output(component_output)
        output_text = _extract_haystack_reply_text(replies[0] if replies else "")

        return LLMResult(generations=[[Generation(text=output_text)]])

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: t.Optional[Callbacks] = None,
    ) -> LLMResult:
        llm_input = self._pipeline_llm_input(prompt, temperature)
        pipeline_output = await self.async_pipeline.run_async(data={"llm": llm_input})
        replies = self._replies_from_output(pipeline_output)
        output_text = _extract_haystack_reply_text(replies[0] if replies else "")

        return LLMResult(generations=[[Generation(text=output_text)]])

    def __repr__(self) -> str:
        try:
            _get_haystack_generator_types()
        except ImportError:
            return f"{self.__class__.__name__}(llm=Unknown(...))"

        return f"{self.__class__.__name__}(llm={_haystack_generator_model_name(self.generator)}(...))"
