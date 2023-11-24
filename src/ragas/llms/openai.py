from __future__ import annotations

import asyncio
import logging
import os
import typing as t
from abc import abstractmethod
from dataclasses import dataclass, field

import openai
from langchain.adapters.openai import convert_message_to_dict
from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.schema import Generation, LLMResult
from openai import AsyncAzureOpenAI, AsyncClient, AsyncOpenAI
from tenacity import (
    RetryCallState,
    before_sleep_log,
    retry,
    retry_base,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ragas.async_utils import run_async_tasks
from ragas.exceptions import AzureOpenAIKeyNotFound, OpenAIKeyNotFound
from ragas.llms.base import RagasLLM
from ragas.llms.langchain import _compute_token_usage_langchain
from ragas.utils import NO_KEY, get_debug_mode

if t.TYPE_CHECKING:
    from langchain.callbacks.base import Callbacks
    from langchain.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

errors = [
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.InternalServerError,
]


def create_base_retry_decorator(
    error_types: t.List[t.Type[BaseException]],
    max_retries: int = 1,
    run_manager: t.Optional[
        t.Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> t.Callable[[t.Any], t.Any]:
    """Create a retry decorator for a given LLM and provided list of error types."""

    log_level = logging.WARNING if get_debug_mode() else logging.DEBUG
    _logging = before_sleep_log(logger, log_level)

    def _before_sleep(retry_state: RetryCallState) -> None:
        _logging(retry_state)
        if run_manager:
            if isinstance(run_manager, AsyncCallbackManagerForLLMRun):
                coro = run_manager.on_retry(retry_state)
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(coro)
                    else:
                        asyncio.run(coro)
                except Exception as e:
                    logger.error(f"Error in on_retry: {e}")
            else:
                run_manager.on_retry(retry_state)
        return None

    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    retry_instance: "retry_base" = retry_if_exception_type(error_types[0])
    for error in error_types[1:]:
        retry_instance = retry_instance | retry_if_exception_type(error)
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=retry_instance,
        before_sleep=_before_sleep,
    )


retry_decorator = create_base_retry_decorator(errors, max_retries=4)


class OpenAIBase(RagasLLM):
    def __init__(self, model: str, _api_key_env_var: str, timeout: int = 60) -> None:
        self.model = model
        self._api_key_env_var = _api_key_env_var
        self.timeout = timeout

        # api key
        key_from_env = os.getenv(self._api_key_env_var, NO_KEY)
        if key_from_env != NO_KEY:
            self.api_key = key_from_env
        else:
            self.api_key = self.api_key
        self._client: AsyncClient

    @abstractmethod
    def _client_init(self) -> AsyncClient:
        ...

    @property
    def llm(self):
        return self

    def create_llm_result(self, response) -> LLMResult:
        """Create the LLMResult from the choices and prompts."""
        if not isinstance(response, dict):
            response = response.model_dump()

        # token Usage
        token_usage = response.get("usage", {})
        llm_output = {
            "token_usage": token_usage,
            "model_name": None,
            "system_fingerprint": response.get("system_fingerprint", ""),
        }

        choices = response["choices"]
        generations = [
            Generation(
                text=choice["message"]["content"],
                generation_info=dict(
                    finish_reason=choice.get("finish_reason"),
                    logprobs=choice.get("logprobs"),
                ),
            )
            for choice in choices
        ]
        llm_output = {"token_usage": token_usage, "model_name": self.model}
        return LLMResult(generations=[generations], llm_output=llm_output)

    def generate(
        self,
        prompts: list[ChatPromptTemplate],
        n: int = 1,
        temperature: float = 0,
        callbacks: t.Optional[Callbacks] = None,
    ) -> t.Any:  # TODO: LLMResult
        llm_results = run_async_tasks(
            [self.agenerate(p, n, temperature, callbacks) for p in prompts]
        )

        generations = [r.generations[0] for r in llm_results]
        llm_output = _compute_token_usage_langchain(llm_results)
        return LLMResult(generations=generations, llm_output=llm_output)

    @retry_decorator
    async def agenerate(
        self,
        prompt: ChatPromptTemplate,
        n: int = 1,
        temperature: float = 0,
        callbacks: t.Optional[Callbacks] = None,
    ) -> LLMResult:
        # TODO: use callbacks for llm generate
        completion = await self._client.chat.completions.create(
            model=self.model,
            messages=[convert_message_to_dict(m) for m in prompt.format_messages()],  # type: ignore
            temperature=temperature,
            n=n,
        )

        return self.create_llm_result(completion)


@dataclass
class OpenAI(OpenAIBase):
    model: str = "gpt-3.5-turbo-16k"
    api_key: str = field(default=NO_KEY, repr=False)
    _api_key_env_var: str = "OPENAI_API_KEY"

    def __post_init__(self):
        super().__init__(model=self.model, _api_key_env_var=self._api_key_env_var)
        self._client_init()

    def _client_init(self):
        self._client = AsyncOpenAI(api_key=self.api_key, timeout=self.timeout)

    def validate_api_key(self):
        # before validating, check if the api key is already set
        api_key = os.getenv(self._api_key_env_var, NO_KEY)
        if api_key != NO_KEY:
            self._client.api_key = api_key
        if self.llm.api_key == NO_KEY:
            os_env_key = os.getenv(self._api_key_env_var, NO_KEY)
            if os_env_key != NO_KEY:
                self.api_key = os_env_key
            else:
                raise OpenAIKeyNotFound


@dataclass
class AzureOpenAI(OpenAIBase):
    azure_endpoint: str
    deployment: str
    api_version: str
    api_key: str = field(default=NO_KEY, repr=False)
    _api_key_env_var: str = "AZURE_OPENAI_API_KEY"

    def __post_init__(self):
        super().__init__(model=self.deployment, _api_key_env_var=self._api_key_env_var)
        self._client_init()

    def _client_init(self):
        self._client = AsyncAzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    def validate_api_key(self):
        if self.llm.api_key == NO_KEY:
            os_env_key = os.getenv(self._api_key_env_var, NO_KEY)
            if os_env_key != NO_KEY:
                self.api_key = os_env_key
            else:
                raise AzureOpenAIKeyNotFound
