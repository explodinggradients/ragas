import typing as t

from ragas.llms import BaseRagasLLM

from .abstract_qa import AbstractQuestionSimulator, ComparativeAbstractQuestionSimulator
from .base import BaseSimulator
from .specific_qa import SpecificQASimulator

SimulatorDistributions = t.List[t.Tuple[BaseSimulator, float]]


def default_simulator_distribution(llm: BaseRagasLLM) -> SimulatorDistributions:
    return [
        (AbstractQuestionSimulator(llm=llm), 0.25),
        (ComparativeAbstractQuestionSimulator(llm=llm), 0.25),
        (SpecificQASimulator(llm=llm), 0.5),
    ]


__all__ = ["AbstractQuestionSimulator", "default_simulator_distribution"]
