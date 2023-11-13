from ragas.llms.base import RagasLLM
from ragas.llms.langchain import LangchainLLM
from ragas.llms.llamaindex import LlamaIndexLLM
from ragas.llms.openai import OpenAI

__all__ = ["RagasLLM", "LangchainLLM", "LlamaIndexLLM", "llm_factory", "OpenAI"]


def llm_factory(model="gpt-3.5-turbo-16k") -> RagasLLM:
    return OpenAI(model=model)
