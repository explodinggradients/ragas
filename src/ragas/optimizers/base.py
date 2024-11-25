import typing as t
from abc import ABC, abstractmethod

from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms.base import BaseRagasLLM
from ragas.metrics.base import MetricWithLLM


class Optimizer(ABC):
    """
    Abstract base class for all optimizers.
    """

    llm: BaseRagasLLM
    embedding: t.Optional[BaseRagasEmbeddings] = None

    @abstractmethod
    def optimize(
        self,
        metric: MetricWithLLM,
        train_data: t.Any,
        config: t.Dict[t.Any, t.Any],
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
