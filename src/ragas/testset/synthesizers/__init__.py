import typing as t

from ragas.llms import BaseRagasLLM

from .abstract_query import (
    AbstractQuerySynthesizer,
    ComparativeAbstractQuerySynthesizer,
)
from .base import BaseSynthesizer
from .base_query import QuerySynthesizer
from .specific_query import SpecificQuerySynthesizer

QueryDistribution = t.List[t.Tuple[BaseSynthesizer, float]]


def default_query_distribution(llm: BaseRagasLLM) -> QueryDistribution:
    """
    Default query distribution for the test set.

    By default, 25% of the queries are generated using `AbstractQuerySynthesizer`,
    25% are generated using `ComparativeAbstractQuerySynthesizer`, and 50% are
    generated using `SpecificQuerySynthesizer`.
    """
    return [
        (AbstractQuerySynthesizer(llm=llm), 0.25),
        (ComparativeAbstractQuerySynthesizer(llm=llm), 0.25),
        (SpecificQuerySynthesizer(llm=llm), 0.5),
    ]


__all__ = [
    "BaseSynthesizer",
    "QuerySynthesizer",
    "AbstractQuerySynthesizer",
    "ComparativeAbstractQuerySynthesizer",
    "SpecificQuerySynthesizer",
    "default_query_distribution",
]
