from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod

from langchain.schema import LLMResult

if t.TYPE_CHECKING:
    from langchain.callbacks.base import Callbacks
    from langchain.prompts import ChatPromptTemplate


class RagasLLM(ABC):
    """
    BaseLLM is the base class for all LLMs. It provides a consistent interface for other
    classes that interact with LLMs like Langchains, LlamaIndex, LiteLLM etc. Handles
    multiple_completions even if not supported by the LLM.

    It currently takes in ChatPromptTemplates and returns LLMResults which are Langchain
    primitives.
    """

    # supports multiple compeletions for the given prompt
    n_completions_supported: bool = False

    @property
    @abstractmethod
    def llm(self) -> t.Any:
        ...

    def validate_api_key(self):
        """
        Validates that the api key is set for the LLM
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompts: list[ChatPromptTemplate],
        n: int = 1,
        temperature: float = 1e-8,
        callbacks: t.Optional[Callbacks] = None,
    ) -> LLMResult:
        ...

    @abstractmethod
    async def agenerate(
        self,
        prompt: ChatPromptTemplate,
        n: int = 1,
        temperature: float = 1e-8,
        callbacks: t.Optional[Callbacks] = None,
    ) -> LLMResult:
        ...
