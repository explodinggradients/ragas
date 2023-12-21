from langchain.chat_models import ChatOpenAI

from ragas.llms.base import BaseRagasLLM, LangchainLLMWrapper
from ragas.llms.llamaindex import LlamaIndexLLM

__all__ = [
    "BaseRagasLLM",
    "LlamaIndexLLM",
    "llm_factory",
]


def llm_factory(model="gpt-3.5-turbo-16k") -> BaseRagasLLM:
    return LangchainLLMWrapper(ChatOpenAI(model=model))
