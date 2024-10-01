import typing as t

from ragas.llms import BaseRagasLLM

from .abstract_query import (
    AbstractQuerySynthesizer,
    ComparativeAbstractQuerySynthesizer,
)
from .base import BaseSynthesizer
from .specific_query import SpecificQuerySynthesizer

QueryDistribution = t.List[t.Tuple[BaseSynthesizer, float]]


def default_query_distribution(llm: BaseRagasLLM) -> QueryDistribution:
    return [
        (AbstractQuerySynthesizer(llm=llm), 0.25),
        (ComparativeAbstractQuerySynthesizer(llm=llm), 0.25),
        (SpecificQuerySynthesizer(llm=llm), 0.5),
    ]


__all__ = ["AbstractQuerySynthesizer", "default_query_distribution"]
