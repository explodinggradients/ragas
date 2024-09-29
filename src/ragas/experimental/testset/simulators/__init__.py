import typing as t

from ragas.llms import BaseRagasLLM

from .abstract_qa import AbstractQuestionSimulator, ComparativeAbstractQuestionSimulator
from .base import BaseSimulator
from .specific_qa import SpecificQASimulator

QuestionTypes = t.List[t.Tuple[BaseSimulator, float]]


def default_scenarios(llm: BaseRagasLLM) -> QuestionTypes:
    return [
        (AbstractQuestionSimulator(llm), 0.25),
        (ComparativeAbstractQuestionSimulator(llm), 0.25),
        (SpecificQASimulator(llm), 0.5),
    ]


__all__ = ["AbstractQuestionSimulator", "default_scenarios"]
