import typing as t

from langchain_core.language_models import BaseLanguageModel

from ragas.llms import BaseRagasLLM, LangchainLLMWrapper, llm_factory
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

    for metric in metrics:
        if metric.llm is None or llm is not None:
            metric.llm = llm_wraper

        if hasattr(metric, "adapt"):
            metric.adapt(language, cache_dir=cache_dir)
            metric.save(cache_dir=cache_dir)
