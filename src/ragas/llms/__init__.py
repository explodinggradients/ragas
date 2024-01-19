from langchain_openai.chat_models import ChatOpenAI

from ragas.llms.base import BaseRagasLLM, LangchainLLMWrapper

__all__ = [
    "BaseRagasLLM",
    "LangchainLLMWrapper",
    "llm_factory",
]


def llm_factory(model="gpt-3.5-turbo-16k") -> BaseRagasLLM:
    return LangchainLLMWrapper(ChatOpenAI(model=model))
