import json
import typing as t
from dataclasses import dataclass

import requests
from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import PromptValue
from ragas.cache import CacheInterface
from ragas.llms import BaseRagasLLM
from ragas.run_config import RunConfig


@dataclass
class FabrixLLMWrapper(BaseRagasLLM):
    """
    A wrapper class for Samsung Fabrix Gauss2 LLM within the Ragas framework.

    Parameters
    ----------
    base_url : str
        The base URL for the Fabrix API endpoint
    llm_id : int
        The LLM ID for the Gauss2 model
    x_openapi_token : str
        The OpenAPI token for authentication
    x_generative_ai_client : str
        The generative AI client token for authentication
    run_config : RunConfig, optional
        Configuration for the run, by default None
    cache : CacheInterface, optional
        A cache instance for storing results, by default None
    """

    base_url: str
    llm_id: int
    x_openapi_token: str
    x_generative_ai_client: str
    run_config: t.Optional[RunConfig] = None
    cache: t.Optional[CacheInterface] = None

    def __post_init__(self):
        super().__post_init__()
        if self.run_config is None:
            self.run_config = RunConfig()
        self.set_run_config(self.run_config)

    def _prepare_request_data(
        self, prompt: str, temperature: float = 1e-8, max_tokens: int = 2048, **kwargs
    ) -> dict:
        """Prepare the request data for Fabrix API."""
        return {
            "llmId": self.llm_id,
            "contents": [prompt],
            "llmConfig": {
                "do_sample": True,
                "max_tokens": max_tokens,
                "return_full_text": False,
                "top_k": 14,
                "top_p": 0.94,
                "temperature": temperature,
                "repetition_penalty": 1.04,
            },
            "isStream": False,
            "systemPrompt": kwargs.get("system_prompt", ""),
        }

    def _parse_response(self, response_data: dict) -> str:
        """Parse the Fabrix API response to extract the generated text."""
        if response_data.get("status") == "SUCCESS":
            return response_data.get("content", "")
        else:
            raise Exception(
                f"Fabrix API error: {response_data.get('responseCode', 'Unknown error')}"
            )

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: t.Optional[t.List[str]] = None,
        callbacks: t.Optional[t.Any] = None,
    ) -> LLMResult:
        """Generate text using Fabrix Gauss2 model synchronously."""

        # Convert PromptValue to string
        prompt_str = prompt.to_string()

        # Prepare request data
        request_data = self._prepare_request_data(
            prompt=prompt_str, temperature=temperature
        )

        # Make API request
        headers = {
            "x-openapi-token": self.x_openapi_token,
            "x-generative-ai-client": self.x_generative_ai_client,
            "Content-Type": "application/json",
        }

        response = requests.post(
            f"{self.base_url}/dev/fssegn/sena_dev_chat_v1/1/openapi/chat/v1/messages",
            headers=headers,
            json=request_data,
            timeout=self.run_config.timeout,
        )

        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

        response_data = response.json()
        generated_text = self._parse_response(response_data)

        # Create LLMResult with the generated text
        generations = [[Generation(text=generated_text)]]
        return LLMResult(generations=generations)

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks: t.Optional[t.Any] = None,
    ) -> LLMResult:
        """Generate text using Fabrix Gauss2 model asynchronously."""

        if temperature is None:
            temperature = self.get_temperature(n)

        # For now, we'll use the synchronous version
        # In a production environment, you might want to use aiohttp for true async
        return self.generate_text(
            prompt=prompt, n=n, temperature=temperature, stop=stop, callbacks=callbacks
        )

    def is_finished(self, response: LLMResult) -> bool:
        """Check if the LLM response is finished."""
        # For Fabrix, we assume the response is always finished
        # since we're not using streaming
        return True

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(llm_id={self.llm_id}, base_url={self.base_url})"
        )
