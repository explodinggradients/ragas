import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

from langchain_core.callbacks import Callbacks

from ragas.llms.base import BaseRagasLLM
from ragas.losses import Loss
from ragas.metrics.base import MetricWithLLM


@dataclass
class Optimizer(ABC):
    """
    Abstract base class for all optimizers.
    """

    metric: t.Optional[MetricWithLLM] = None
    llm: t.Optional[BaseRagasLLM] = None

    @abstractmethod
    def optimize(
        self,
        train_data: t.Any,
        loss: Loss,
        config: t.Dict[t.Any, t.Any],
        callbacks: Callbacks,
    ) -> MetricWithLLM:
        """
        Optimizes the prompts for the given metric.

        Parameters
        ----------
        metric : MetricWithLLM
            The metric to optimize.
        train_data : Any
            The training data.
        config : InstructionConfig
            The training configuration.

        Returns
        -------
        MetricWithLLM
            The optimized metric.
        """
        pass
