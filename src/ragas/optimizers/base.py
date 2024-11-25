import typing as t
from abc import ABC, abstractmethod

from ragas.config import InstructionConfig
from ragas.metrics.base import MetricWithLLM


class Optimizer(ABC):
    """
    Abstract base class for all optimizers.
    """

    @abstractmethod
    def optimize(
        self,
        metric: MetricWithLLM,
        train_data: t.Any,
        config: InstructionConfig,
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
