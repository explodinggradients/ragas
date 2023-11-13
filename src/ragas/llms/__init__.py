from ragas.llms.base import BaseRagasLLM
from ragas.llms.langchain import LangchainLLM
from ragas.llms.llamaindex import LlamaIndexLLM
from ragas.llms.openai import OpenAI

__all__ = ["BaseRagasLLM", "LangchainLLM", "LlamaIndexLLM", "llm_factory", "OpenAI"]


def llm_factory() -> BaseRagasLLM:
    return OpenAI()
