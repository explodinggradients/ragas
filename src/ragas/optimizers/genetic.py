import typing as t

from langchain_core.callbacks import Callbacks

from ragas.metrics.base import MetricWithLLM
from ragas.optimizers.base import Optimizer


class GeneticOptimizer(Optimizer):
    """
    A genetic algorithm optimizer that balances exploration and exploitation.
    """

    def optimize(
        self,
        metric: MetricWithLLM,
        train_data: t.Any,
        config: t.Dict[t.Any, t.Any],
        callbacks: Callbacks,
    ) -> MetricWithLLM:

        # max_steps = config.get("max_steps", 100)

        return metric
