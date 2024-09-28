import typing as t

from ragas.llms import BaseRagasLLM

from .abstract import AbstractQuestionSimulator
from .base import BaseSimulator

QuestionTypes = t.List[t.Tuple[BaseSimulator, float]]


def default_scenarios(llm: BaseRagasLLM) -> QuestionTypes:
    return [(AbstractQuestionSimulator(llm), 1.0)]


__all__ = ["AbstractQuestionSimulator", "default_scenarios"]
