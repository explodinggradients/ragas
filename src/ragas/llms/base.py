from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import instructor
from langchain_community.chat_models.vertexai import ChatVertexAI
from langchain_community.llms import VertexAI
from langchain_core.language_models import BaseLanguageModel
from langchain_core.outputs import ChatGeneration, Generation, LLMResult
from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain_openai.llms import AzureOpenAI, OpenAI
from langchain_openai.llms.base import BaseOpenAI
from pydantic import BaseModel

from ragas.cache import CacheInterface, cacher
from ragas.exceptions import LLMDidNotFinishException
from ragas.integrations.helicone import helicone_config
from ragas.run_config import RunConfig, add_async_retry

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks
    from langchain_core.messages import BaseMessage
    from langchain_core.prompt_values import PromptValue
    from llama_index.core.base.llms.base import BaseLLM


logger = logging.getLogger(__name__)

# TypeVar for Instructor LLM response models
InstructorTypeVar = t.TypeVar("T", bound=BaseModel)  # type: ignore

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
    run_config: RunConfig = field(default_factory=RunConfig, repr=False)
    multiple_completion_supported: bool = field(default=False, repr=False)
    cache: t.Optional[CacheInterface] = field(default=None, repr=False)

    def __post_init__(self):
        # If a cache_backend is provided, wrap the implementation methods at construction time.
        if self.cache is not None:
            self.generate_text = cacher(cache_backend=self.cache)(self.generate_text)
            self.agenerate_text = cacher(cache_backend=self.cache)(self.agenerate_text)

    def set_run_config(self, run_config: RunConfig):
        self.run_config = run_config

    def get_temperature(self, n: int) -> float:
        """Return the temperature to use for completion based on n."""
        return 0.3 if n > 1 else 0.01

    @abstractmethod
    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult: ...

    @abstractmethod
    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult: ...

    @abstractmethod
    def is_finished(self, response: LLMResult) -> bool:
        """Check if the LLM response is finished/complete."""
        ...

    async def generate(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = 0.01,
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
        langchain_llm: BaseLanguageModel[BaseMessage],
        run_config: t.Optional[RunConfig] = None,
        is_finished_parser: t.Optional[t.Callable[[LLMResult], bool]] = None,
        cache: t.Optional[CacheInterface] = None,
        bypass_temperature: bool = False,
    ):
        super().__init__(cache=cache)
        self.langchain_llm = langchain_llm
        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)
        self.is_finished_parser = is_finished_parser
        # Certain LLMs (e.g., OpenAI o1 series) do not support temperature
        self.bypass_temperature = bypass_temperature

    def is_finished(self, response: LLMResult) -> bool:
        """
        Parse the response to check if the LLM finished by checking the finish_reason
        or stop_reason. Supports OpenAI and Vertex AI models.
        """
        if self.is_finished_parser is not None:
            return self.is_finished_parser(response)
        # if no parser is provided default to our own

        is_finished_list = []
        for g in response.flatten():
            resp = g.generations[0][0]
            if resp.generation_info is not None:
                # generation_info is provided - so we parse that
                finish_reason = resp.generation_info.get("finish_reason")
                if finish_reason is not None:
                    # OpenAI uses "stop"
                    # Vertex AI uses "STOP" or "MAX_TOKENS"
                    # WatsonX AI uses "eos_token"
                    is_finished_list.append(
                        finish_reason in ["stop", "STOP", "MAX_TOKENS", "eos_token"]
                    )

                # provied more conditions here
                # https://github.com/explodinggradients/ragas/issues/1548

            # if generation_info is empty, we parse the response_metadata
            # this is less reliable

            elif (
                isinstance(resp, ChatGeneration)
                and t.cast(ChatGeneration, resp).message is not None
            ):
                resp_message: BaseMessage = t.cast(ChatGeneration, resp).message
                if resp_message.response_metadata.get("finish_reason") is not None:
                    finish_reason = resp_message.response_metadata.get("finish_reason")
                    is_finished_list.append(
                        finish_reason in ["stop", "STOP", "MAX_TOKENS", "eos_token"]
                    )
                elif resp_message.response_metadata.get("stop_reason") is not None:
                    stop_reason = resp_message.response_metadata.get("stop_reason")
                    is_finished_list.append(
                        stop_reason
                        in ["end_turn", "stop", "STOP", "MAX_TOKENS", "eos_token"]
                    )
            # default to True
            else:
                is_finished_list.append(True)
        return all(is_finished_list)

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        # figure out the temperature to set
        old_temperature: float | None = None
        if temperature is None:
            temperature = self.get_temperature(n=n)
        if hasattr(self.langchain_llm, "temperature"):
            self.langchain_llm.temperature = temperature  # type: ignore
            old_temperature = temperature

        if is_multiple_completion_supported(self.langchain_llm):
            result = self.langchain_llm.generate_prompt(
                prompts=[prompt],
                n=n,
                stop=stop,
                callbacks=callbacks,
            )
        else:
            result = self.langchain_llm.generate_prompt(
                prompts=[prompt] * n,
                stop=stop,
                callbacks=callbacks,
            )
            # make LLMResult.generation appear as if it was n_completions
            # note that LLMResult.runs is still a list that represents each run
            generations = [[g[0] for g in result.generations]]
            result.generations = generations

        # reset the temperature to the original value
        if old_temperature is not None:
            self.langchain_llm.temperature = old_temperature  # type: ignore

        return result

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        # handle temperature
        old_temperature: float | None = None
        if temperature is None:
            temperature = self.get_temperature(n=n)
        if hasattr(self.langchain_llm, "temperature") and not self.bypass_temperature:
            self.langchain_llm.temperature = temperature  # type: ignore
            old_temperature = temperature

        # handle n
        if hasattr(self.langchain_llm, "n"):
            self.langchain_llm.n = n  # type: ignore
            result = await self.langchain_llm.agenerate_prompt(
                prompts=[prompt],
                stop=stop,
                callbacks=callbacks,
            )
        else:
            result = await self.langchain_llm.agenerate_prompt(
                prompts=[prompt] * n,
                stop=stop,
                callbacks=callbacks,
            )
            # make LLMResult.generation appear as if it was n_completions
            # note that LLMResult.runs is still a list that represents each run
            generations = [[g[0] for g in result.generations]]
            result.generations = generations

        # reset the temperature to the original value
        if old_temperature is not None:
            self.langchain_llm.temperature = old_temperature  # type: ignore

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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(langchain_llm={self.langchain_llm.__class__.__name__}(...))"


class LlamaIndexLLMWrapper(BaseRagasLLM):
    """
    A Adaptor for LlamaIndex LLMs
    """

    def __init__(
        self,
        llm: BaseLLM,
        run_config: t.Optional[RunConfig] = None,
        cache: t.Optional[CacheInterface] = None,
        bypass_temperature: bool = False,
    ):
        super().__init__(cache=cache)
        self.llm = llm
        # Certain LLMs (e.g., OpenAI o1 series) do not support temperature
        self.bypass_temperature = bypass_temperature

        try:
            self._signature = type(self.llm).__name__.lower()
        except AttributeError:
            self._signature = ""

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
        if temperature != 0.01:
            logger.info("temperature kwarg passed to LlamaIndex LLM")
        if stop is not None:
            logger.info("stop kwarg passed to LlamaIndex LLM")
        if callbacks is not None:
            logger.info(
                "callbacks not supported for LlamaIndex LLMs, ignoring callbacks"
            )
        if self._signature in ["anthropic", "bedrock"]:
            return {"temperature": temperature}
        else:
            return {
                "n": n,
                "temperature": temperature,
                "stop": stop,
            }

    def is_finished(self, response: LLMResult) -> bool:
        return True

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.01,
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

        if self.bypass_temperature:
            kwargs.pop("temperature", None)

        li_response = await self.llm.acomplete(prompt.to_string(), **kwargs)

        return LLMResult(generations=[[Generation(text=li_response.text)]])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(llm={self.llm.__class__.__name__}(...))"


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


# Experimental LLM classes migrated from ragas.experimental.llms


class InstructorBaseRagasLLM(ABC):
    """Base class for LLMs using the Instructor library pattern."""

    @abstractmethod
    def generate(
        self, prompt: str, response_model: t.Type[InstructorTypeVar]
    ) -> InstructorTypeVar:
        """Generate a response using the configured LLM.

        For async clients, this will run the async method in the appropriate event loop.
        """

    @abstractmethod
    async def agenerate(
        self, prompt: str, response_model: t.Type[InstructorTypeVar]
    ) -> InstructorTypeVar:
        """Asynchronously generate a response using the configured LLM."""


class InstructorLLM(InstructorBaseRagasLLM):
    """LLM wrapper using the Instructor library for structured outputs."""

    def __init__(self, client: t.Any, model: str, provider: str, **model_args):
        self.client = client
        self.model = model
        self.provider = provider
        self.model_args = model_args or {}
        # Check if client is async-capable at initialization
        self.is_async = self._check_client_async()

    def _check_client_async(self) -> bool:
        """Determine if the client is async-capable."""
        try:
            # Check if this is an async client by checking for a coroutine method
            if hasattr(self.client.chat.completions, "create"):
                return inspect.iscoroutinefunction(self.client.chat.completions.create)
            return False
        except (AttributeError, TypeError):
            return False

    def _run_async_in_current_loop(self, coro: t.Awaitable[t.Any]) -> t.Any:
        """Run an async coroutine in the current event loop if possible.

        This handles Jupyter environments correctly by using a separate thread
        when a running event loop is detected.
        """
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()

            if loop.is_running():
                # If the loop is already running (like in Jupyter notebooks),
                # we run the coroutine in a separate thread with its own event loop
                result_container: t.Dict[str, t.Any] = {
                    "result": None,
                    "exception": None,
                }

                def run_in_thread():
                    # Create a new event loop for this thread
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        # Run the coroutine in this thread's event loop
                        result_container["result"] = new_loop.run_until_complete(coro)
                    except Exception as e:
                        # Capture any exceptions to re-raise in the main thread
                        result_container["exception"] = e
                    finally:
                        # Clean up the event loop
                        new_loop.close()

                # Start the thread and wait for it to complete
                thread = threading.Thread(target=run_in_thread)
                thread.start()
                thread.join()

                # Re-raise any exceptions that occurred in the thread
                if result_container["exception"]:
                    raise result_container["exception"]

                return result_container["result"]
            else:
                # Standard case - event loop exists but isn't running
                return loop.run_until_complete(coro)

        except RuntimeError:
            # If we get a runtime error about no event loop, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                # Clean up
                loop.close()
                asyncio.set_event_loop(None)

    def generate(
        self, prompt: str, response_model: t.Type[InstructorTypeVar]
    ) -> InstructorTypeVar:
        """Generate a response using the configured LLM.

        For async clients, this will run the async method in the appropriate event loop.
        """
        messages = [{"role": "user", "content": prompt}]

        # If client is async, use the appropriate method to run it
        if self.is_async:
            return self._run_async_in_current_loop(
                self.agenerate(prompt, response_model)
            )
        else:
            # Regular sync client, just call the method directly
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_model=response_model,
                **self.model_args,
            )

    async def agenerate(
        self, prompt: str, response_model: t.Type[InstructorTypeVar]
    ) -> InstructorTypeVar:
        """Asynchronously generate a response using the configured LLM."""
        messages = [{"role": "user", "content": prompt}]

        # If client is not async, raise a helpful error
        if not self.is_async:
            raise TypeError(
                "Cannot use agenerate() with a synchronous client. Use generate() instead."
            )

        # Regular async client, call the method directly
        return await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_model=response_model,
            **self.model_args,
        )

    def _get_client_info(self) -> str:
        """Get client type and async status information."""
        client_type = self.client.__class__.__name__
        async_status = "async" if self.is_async else "sync"
        return f"<{client_type}:{async_status}>"

    def _get_key_config(self) -> str:
        """Get key configuration parameters as a string."""
        config_parts = []

        # Show important model arguments
        important_args = [
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
        ]
        for arg in important_args:
            if arg in self.model_args:
                config_parts.append(f"{arg}={self.model_args[arg]}")

        # Show count of other args if there are any
        other_args = len([k for k in self.model_args.keys() if k not in important_args])
        if other_args > 0:
            config_parts.append(f"+{other_args} more")

        return ", ".join(config_parts)

    def __repr__(self) -> str:
        """Return a detailed string representation of the LLM."""
        client_info = self._get_client_info()
        key_config = self._get_key_config()

        base_repr = f"InstructorLLM(provider='{self.provider}', model='{self.model}', client={client_info}"

        if key_config:
            base_repr += f", {key_config}"

        base_repr += ")"
        return base_repr

    __str__ = __repr__


def instructor_llm_factory(
    provider: str,
    model: t.Optional[str] = None,
    client: t.Optional[t.Any] = None,
    **kwargs: t.Any,
) -> InstructorBaseRagasLLM:
    """
    Factory function to create an InstructorLLM instance based on the provider.

    Args:
        provider (str): The name of the LLM provider or provider/model string
                       (e.g., "openai", "openai/gpt-4").
        model (str, optional): The model name to use for generation.
        client (Any, optional): Pre-initialized client for the provider.
        **kwargs: Additional arguments for the LLM (model_args).

    Returns:
        InstructorBaseRagasLLM: An instance of the specified LLM provider.

    Examples:
        # OpenAI with separate parameters
        llm = instructor_llm_factory("openai", "gpt-4", client=openai_client)

        # OpenAI with provider/model string
        llm = instructor_llm_factory("openai/gpt-4", client=openai_client)

        # Anthropic
        llm = instructor_llm_factory("anthropic", "claude-3-sonnet-20240229", client=anthropic_client)

        # Cohere
        llm = instructor_llm_factory("cohere", "command-r-plus", client=cohere_client)

        # Gemini
        llm = instructor_llm_factory("gemini", "gemini-pro", client=gemini_client)

        # LiteLLM (supports 100+ models)
        llm = instructor_llm_factory("litellm", "gpt-4", client=litellm_client)

    Raises:
        ValueError: If provider is unsupported or required parameters are missing.
    """
    # Handle provider/model string format
    if "/" in provider and model is None:
        provider_name, model_name = provider.split("/", 1)
        provider = provider_name
        model = model_name

    if not model:
        raise ValueError(
            "Model name is required. Either provide it as a separate parameter "
            "or use provider/model format (e.g., 'openai/gpt-4')"
        )

    def _initialize_client(provider: str, client: t.Any) -> t.Any:
        """Initialize the instructor-patched client for the given provider."""
        if not client:
            raise ValueError(f"{provider.title()} provider requires a client instance")

        provider_lower = provider.lower()

        if provider_lower == "openai":
            return instructor.from_openai(client)
        elif provider_lower == "anthropic":
            return instructor.from_anthropic(client)
        elif provider_lower == "cohere":
            return instructor.from_cohere(client)
        elif provider_lower == "gemini":
            return instructor.from_gemini(client)
        elif provider_lower == "litellm":
            return instructor.from_litellm(client)
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported providers: openai, anthropic, cohere, gemini, litellm"
            )

    instructor_patched_client = _initialize_client(provider=provider, client=client)
    return InstructorLLM(
        client=instructor_patched_client, model=model, provider=provider, **kwargs
    )
