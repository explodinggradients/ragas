from __future__ import annotations

import logging
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from langchain_community.chat_models.vertexai import ChatVertexAI
from langchain_community.llms import VertexAI
from langchain_core.language_models import BaseLanguageModel
from langchain_core.outputs import ChatGeneration, Generation, LLMResult
from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain_openai.llms import AzureOpenAI, OpenAI
from langchain_openai.llms.base import BaseOpenAI

from ragas.exceptions import LLMDidNotFinishException
from ragas.integrations.helicone import helicone_config
from ragas.run_config import RunConfig, add_async_retry

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks
    from langchain_core.messages import BaseMessage
    from langchain_core.prompt_values import PromptValue
    from llama_index.core.base.llms.base import BaseLLM

logger = logging.getLogger(__name__)

MULTIPLE_COMPLETION_SUPPORTED = [
    OpenAI,
    ChatOpenAI,
    AzureOpenAI,
    AzureChatOpenAI,
    ChatVertexAI,
    VertexAI,
]


def is_multiple_completion_supported(llm: BaseLanguageModel) -> bool:
    """Return whether the given LLM supports n-completion."""
    for llm_type in MULTIPLE_COMPLETION_SUPPORTED:
        if isinstance(llm, llm_type):
            return True
    return False


@dataclass
class BaseRagasLLM(ABC):
    run_config: RunConfig = field(default_factory=RunConfig)
    multiple_completion_supported: bool = False

    def set_run_config(self, run_config: RunConfig):
        self.run_config = run_config

    def get_temperature(self, n: int) -> float:
        """Return the temperature to use for completion based on n."""
        return 0.3 if n > 1 else 1e-8

    def is_finished(self, response: LLMResult) -> bool:
        logger.warning(
            f"is_finished not implemented for {self.__class__.__name__}. Will default to True."
        )
        return True

    @abstractmethod
    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult: ...

    @abstractmethod
    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult: ...

    async def generate(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        """Generate text using the given event loop."""

        if temperature is None:
            temperature = self.get_temperature(n)

        agenerate_text_with_retry = add_async_retry(
            self.agenerate_text, self.run_config
        )
        result = await agenerate_text_with_retry(
            prompt=prompt,
            n=n,
            temperature=temperature,
            stop=stop,
            callbacks=callbacks,
        )

        # check there are no max_token issues
        if not self.is_finished(result):
            raise LLMDidNotFinishException()
        return result


class LangchainLLMWrapper(BaseRagasLLM):
    """
    A simple base class for RagasLLMs that is based on Langchain's BaseLanguageModel
    interface. it implements 2 functions:
    - generate_text: for generating text from a given PromptValue
    - agenerate_text: for generating text from a given PromptValue asynchronously
    """

    def __init__(
        self,
        langchain_llm: BaseLanguageModel,
        run_config: t.Optional[RunConfig] = None,
        is_finished_parser: t.Optional[t.Callable[[LLMResult], bool]] = None,
    ):
        self.langchain_llm = langchain_llm
        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)
        self.is_finished_parser = is_finished_parser

    def is_finished(self, response: LLMResult) -> bool:
        """
        Parse the response to check if the LLM finished by checking the finish_reason
        or stop_reason.
        """
        if self.is_finished_parser is not None:
            return self.is_finished_parser(response)
        # if no parser is provided default to our own

        is_finished_list = []
        for g in response.flatten():
            resp = g.generations[0][0]
            if resp.generation_info is not None:
                # generation_info is provided - so we parse that

                # OpenAI uses "stop" to indicate that the generation is finished
                # and is stored in 'finish_reason' key in generation_info
                if resp.generation_info.get("finish_reason") is not None:
                    is_finished_list.append(
                        resp.generation_info.get("finish_reason") == "stop"
                    )
                # provied more conditions here
                # https://github.com/explodinggradients/ragas/issues/1548

            # if generation_info is empty, we parse the response_metadata
            # this is less reliable
            elif t.cast(ChatGeneration, resp).message is not None:
                resp_message: BaseMessage = t.cast(ChatGeneration, resp).message
                if resp_message.response_metadata.get("finish_reason") is not None:
                    is_finished_list.append(
                        resp_message.response_metadata.get("finish_reason") == "stop"
                    )
                elif resp_message.response_metadata.get("stop_reason") is not None:
                    is_finished_list.append(
                        resp_message.response_metadata.get("stop_reason") == "end_turn"
                    )
            # default to True
            else:
                is_finished_list.append(True)
        return all(is_finished_list)

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        # figure out the temperature to set
        if temperature is None:
            temperature = self.get_temperature(n=n)

        if is_multiple_completion_supported(self.langchain_llm):
            return self.langchain_llm.generate_prompt(
                prompts=[prompt],
                n=n,
                temperature=temperature,
                stop=stop,
                callbacks=callbacks,
            )
        else:
            result = self.langchain_llm.generate_prompt(
                prompts=[prompt] * n,
                temperature=temperature,
                stop=stop,
                callbacks=callbacks,
            )
            # make LLMResult.generation appear as if it was n_completions
            # note that LLMResult.runs is still a list that represents each run
            generations = [[g[0] for g in result.generations]]
            result.generations = generations
            return result

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        if temperature is None:
            temperature = self.get_temperature(n=n)

        if is_multiple_completion_supported(self.langchain_llm):
            return await self.langchain_llm.agenerate_prompt(
                prompts=[prompt],
                n=n,
                temperature=temperature,
                stop=stop,
                callbacks=callbacks,
            )
        else:
            result = await self.langchain_llm.agenerate_prompt(
                prompts=[prompt] * n,
                temperature=temperature,
                stop=stop,
                callbacks=callbacks,
            )
            # make LLMResult.generation appear as if it was n_completions
            # note that LLMResult.runs is still a list that represents each run
            generations = [[g[0] for g in result.generations]]
            result.generations = generations
            return result

    def set_run_config(self, run_config: RunConfig):
        self.run_config = run_config

        # configure if using OpenAI API
        if isinstance(self.langchain_llm, BaseOpenAI) or isinstance(
            self.langchain_llm, ChatOpenAI
        ):
            try:
                from openai import RateLimitError
            except ImportError:
                raise ImportError(
                    "openai.error.RateLimitError not found. Please install openai package as `pip install openai`"
                )
            self.langchain_llm.request_timeout = run_config.timeout
            self.run_config.exception_types = RateLimitError


class LlamaIndexLLMWrapper(BaseRagasLLM):
    """
    A Adaptor for LlamaIndex LLMs
    """

    def __init__(
        self,
        llm: BaseLLM,
        run_config: t.Optional[RunConfig] = None,
    ):
        self.llm = llm

        self._signature = ""
        if type(self.llm).__name__.lower() == "bedrock":
            self._signature = "bedrock"
        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)

    def check_args(
        self,
        n: int,
        temperature: float,
        stop: t.Optional[t.List[str]],
        callbacks: Callbacks,
    ) -> dict[str, t.Any]:
        if n != 1:
            logger.warning("n values greater than 1 not support for LlamaIndex LLMs")
        if temperature != 1e-8:
            logger.info("temperature kwarg passed to LlamaIndex LLM")
        if stop is not None:
            logger.info("stop kwarg passed to LlamaIndex LLM")
        if callbacks is not None:
            logger.info(
                "callbacks not supported for LlamaIndex LLMs, ignoring callbacks"
            )
        if self._signature == "bedrock":
            return {"temperature": temperature}
        else:
            return {
                "n": n,
                "temperature": temperature,
                "stop": stop,
            }

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        kwargs = self.check_args(n, temperature, stop, callbacks)
        li_response = self.llm.complete(prompt.to_string(), **kwargs)

        return LLMResult(generations=[[Generation(text=li_response.text)]])

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        if temperature is None:
            temperature = self.get_temperature(n)

        kwargs = self.check_args(n, temperature, stop, callbacks)
        li_response = await self.llm.acomplete(prompt.to_string(), **kwargs)

        return LLMResult(generations=[[Generation(text=li_response.text)]])


def llm_factory(
    model: str = "gpt-4o-mini",
    run_config: t.Optional[RunConfig] = None,
    default_headers: t.Optional[t.Dict[str, str]] = None,
    base_url: t.Optional[str] = None,
) -> BaseRagasLLM:
    """
    Create and return a BaseRagasLLM instance. Used for running default LLMs used
    in Ragas (OpenAI).

    Parameters
    ----------
    model : str, optional
        The name of the model to use, by default "gpt-4o-mini".
    run_config : RunConfig, optional
        Configuration for the run, by default None.
    default_headers : dict of str, optional
        Default headers to be used in API requests, by default None.
    base_url : str, optional
        Base URL for the API, by default None.

    Returns
    -------
    BaseRagasLLM
        An instance of BaseRagasLLM configured with the specified parameters.
    """
    timeout = None
    if run_config is not None:
        timeout = run_config.timeout

    # if helicone is enabled, use the helicone
    if helicone_config.is_enabled:
        default_headers = helicone_config.default_headers()
        base_url = helicone_config.base_url

    openai_model = ChatOpenAI(
        model=model, timeout=timeout, default_headers=default_headers, base_url=base_url
    )
    return LangchainLLMWrapper(openai_model, run_config)
