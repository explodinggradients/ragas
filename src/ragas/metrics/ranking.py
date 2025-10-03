"""Base class for ranking metrics"""

__all__ = ["ranking_metric", "RankingMetric"]

import typing as t
from dataclasses import dataclass

from pydantic import Field

if t.TYPE_CHECKING:
    from ragas.metrics.base import EmbeddingModelType

from .base import SimpleLLMMetric
from .decorator import RankingMetricProtocol, create_metric_decorator
from .validators import RankingValidator


@dataclass(repr=False)
class RankingMetric(SimpleLLMMetric, RankingValidator):
    """
    Metric for evaluations that produce ranked lists of items.

    This class is used for metrics that output ordered lists, such as
    ranking search results, prioritizing features, or ordering responses
    by relevance.

    Attributes
    ----------
    allowed_values : int
        Expected number of items in the ranking list. Default is 2.

    Examples
    --------
    >>> from ragas.metrics import RankingMetric
    >>> from ragas.llms import LangchainLLMWrapper
    >>> from langchain_openai import ChatOpenAI
    >>>
    >>> # Create a ranking metric that returns top 3 items
    >>> llm = LangchainLLMWrapper(ChatOpenAI())
    >>> metric = RankingMetric(
    ...     name="relevance_ranking",
    ...     llm=llm,
    ...     allowed_values=3
    ... )
    """

    allowed_values: int = 2

    def __post_init__(self):
        super().__post_init__()
        # Use the factory to create and mark the model as auto-generated
        from ragas.metrics.base import create_auto_response_model

        self._response_model = create_auto_response_model(
            "RankingResponseModel",
            reason=(str, Field(..., description="Reasoning for the ranking")),
            value=(t.List[str], Field(..., description="List of ranked items")),
        )

    def get_correlation(
        self, gold_labels: t.List[str], predictions: t.List[str]
    ) -> float:
        """
        Calculate the correlation between gold labels and predictions.
        This is a placeholder method and should be implemented based on the specific metric.
        """
        try:
            from sklearn.metrics import cohen_kappa_score
        except ImportError:
            raise ImportError(
                "scikit-learn is required for correlation calculation. "
                "Please install it with `pip install scikit-learn`."
            )

        kappa_scores = []
        for gold_item, prediction in zip(gold_labels, predictions):
            kappa = cohen_kappa_score(gold_item, prediction, weights="quadratic")
            kappa_scores.append(kappa)

        return sum(kappa_scores) / len(kappa_scores) if kappa_scores else 0.0

    @classmethod
    def load(
        cls, path: str, embedding_model: t.Optional["EmbeddingModelType"] = None
    ) -> "RankingMetric":
        """
        Load a RankingMetric from a JSON file.

        Parameters:
        -----------
        path : str
            File path to load from. Supports .gz compressed files.
        embedding_model : Optional[Any]
            Embedding model for DynamicFewShotPrompt. Required if the original used one.

        Returns:
        --------
        RankingMetric
            Loaded metric instance

        Raises:
        -------
        ValueError
            If file cannot be loaded or is not a RankingMetric
        """
        # Validate metric type before loading
        cls._validate_metric_type(path)

        # Load using parent class method
        metric = super().load(path, embedding_model=embedding_model)

        # Additional type check for safety
        if not isinstance(metric, cls):
            raise ValueError(f"Loaded metric is not a {cls.__name__}")

        return metric


def ranking_metric(
    *,
    name: t.Optional[str] = None,
    allowed_values: t.Optional[int] = None,
    **metric_params: t.Any,
) -> t.Callable[[t.Callable[..., t.Any]], RankingMetricProtocol]:
    """
    Decorator for creating ranking/ordering metrics.

    This decorator transforms a regular function into a RankingMetric instance
    that outputs ordered lists of items.

    Parameters
    ----------
    name : str, optional
        Name for the metric. If not provided, uses the function name.
    allowed_values : int, optional
        Expected number of items in the ranking list. Default is 2.
    **metric_params : Any
        Additional parameters to pass to the metric initialization.

    Returns
    -------
    Callable[[Callable[..., Any]], RankingMetricProtocol]
        A decorator that transforms a function into a RankingMetric instance.

    Examples
    --------
    >>> from ragas.metrics import ranking_metric
    >>>
    >>> @ranking_metric(name="priority_ranker", allowed_values=3)
    >>> def rank_by_urgency(user_input: str, responses: list) -> list:
    ...     '''Rank responses by urgency keywords.'''
    ...     urgency_keywords = ["urgent", "asap", "critical"]
    ...     scored = []
    ...     for resp in responses:
    ...         score = sum(kw in resp.lower() for kw in urgency_keywords)
    ...         scored.append((score, resp))
    ...     # Sort by score descending and return top items
    ...     ranked = sorted(scored, key=lambda x: x[0], reverse=True)
    ...     return [item[1] for item in ranked[:3]]
    >>>
    >>> result = rank_by_urgency(
    ...     user_input="What should I do first?",
    ...     responses=["This is urgent", "Take your time", "Critical issue!"]
    ... )
    >>> print(result.value)  # Ranked list of responses
    """
    if allowed_values is None:
        allowed_values = 2

    decorator_factory = create_metric_decorator()
    return decorator_factory(name=name, allowed_values=allowed_values, **metric_params)  # type: ignore[return-value]
