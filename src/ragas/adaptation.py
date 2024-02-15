import typing as t

from langchain_core.language_models import BaseLanguageModel

from ragas.llms import llm_factory
from ragas.llms.base import BaseRagasLLM, LangchainLLMWrapper
from ragas.metrics.base import MetricWithLLM


def adapt(
    metrics: t.List[MetricWithLLM],
    language: str,
    llm: t.Optional[BaseRagasLLM] = None,
    cache_dir: t.Optional[str] = None,
) -> None:
    """
    Adapt the metric to a different language.
    """

    llm_wraper = None

    if llm is None:
        llm_wraper = llm_factory()
    elif isinstance(llm, BaseLanguageModel):
        llm_wraper = LangchainLLMWrapper(llm)
    else:
        raise ValueError("llm must be either None or a BaseLanguageModel")

    for metric in metrics:
        metric_llm = metric.llm

        if metric_llm is None or llm is not None:
            metric.llm = llm_wraper

        if hasattr(metric, "adapt"):
            metric.adapt(language, cache_dir=cache_dir)
            metric.save(cache_dir=cache_dir)
            metric.llm = metric_llm
