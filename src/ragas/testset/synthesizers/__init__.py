import typing as t

from ragas.llms import BaseRagasLLM

from .base import BaseSynthesizer

QueryDistribution = t.List[t.Tuple[BaseSynthesizer, float]]


def default_query_distribution(llm: BaseRagasLLM) -> QueryDistribution:
    """ """
    return []


__all__ = [
    "BaseSynthesizer",
    "default_query_distribution",
]
