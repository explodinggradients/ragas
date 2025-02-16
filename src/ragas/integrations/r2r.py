from __future__ import annotations

import logging
import typing as t
import warnings

from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.evaluation import evaluate as ragas_evaluate
from ragas.executor import Executor
from ragas.run_config import RunConfig

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks
    from r2r import R2RAsyncClient

    from ragas.cost import TokenUsageParser
    from ragas.embeddings import BaseRagasEmbeddings
    from ragas.evaluation import EvaluationResult
    from ragas.llms import BaseRagasLLM
    from ragas.metrics.base import Metric


logger = logging.getLogger(__name__)


def _process_search_results(search_results: t.Dict[str, t.List]) -> t.List[str]:
    """
    Extracts relevant text from search results while issuing warnings for unsupported result types.

    Parameters
    ----------
    search_results : Dict[str, List]
        A r2r result object of an aggregate search operation.

    Returns
    -------
    List[str]
        A list of extracted text from aggregate search result.
    """
    retrieved_contexts = []

    for key in ["graph_search_results", "context_document_results"]:
        if search_results.get(key) and len(search_results[key]) > 0:
            warnings.warn(
                f"{key} are not included in the aggregated `retrieved_context` for Ragas evaluations."
            )

    for result in search_results.get("chunk_search_results", []):
        text = result.get("text")
        if text:
            retrieved_contexts.append(text)

    for result in search_results.get("web_search_results", []):
        text = result.get("snippet")
        if text:
            retrieved_contexts.append(text)

    return retrieved_contexts


def evaluate(
    r2r_client: R2RAsyncClient,
    dataset: EvaluationDataset,
    metrics: list[Metric],
    search_settings: t.Optional[t.Dict[str, t.Any]] = None,
    rag_generation_config: t.Optional[t.Dict[str, t.Any]] = None,
    search_mode: t.Optional[str] = "custom",
    task_prompt_override: t.Optional[str] = None,
    include_title_if_available: t.Optional[bool] = False,
    llm: t.Optional[BaseRagasLLM] = None,
    embeddings: t.Optional[BaseRagasEmbeddings] = None,
    callbacks: t.Optional[Callbacks] = None,
    run_config: t.Optional[RunConfig] = None,
    batch_size: t.Optional[int] = None,
    token_usage_parser: t.Optional[TokenUsageParser] = None,
    raise_exceptions: bool = False,
    column_map: t.Optional[t.Dict[str, str]] = None,
    show_progress: bool = True,
) -> EvaluationResult:
    column_map = column_map or {}

    # validate and transform dataset
    if dataset is None or not isinstance(dataset, EvaluationDataset):
        raise ValueError("Please provide a dataset that is of type EvaluationDataset")

    exec = Executor(
        desc="Querying Client",
        keep_progress_bar=True,
        show_progress=show_progress,
        raise_exceptions=raise_exceptions,
        run_config=run_config,
        batch_size=batch_size,
    )

    # check if multi-turn
    if dataset.is_multi_turn():
        raise NotImplementedError(
            "Multi-turn evaluation is not implemented yet. Please do raise an issue on GitHub if you need this feature and we will prioritize it"
        )
    samples = t.cast(t.List[SingleTurnSample], dataset.samples)

    # get query and make jobs
    queries = [sample.user_input for sample in samples]
    for i, q in enumerate(queries):
        exec.submit(
            r2r_client.retrieval.rag,
            query=q,
            rag_generation_config=rag_generation_config,
            search_mode=search_mode,
            search_settings=search_settings,
            task_prompt_override=task_prompt_override,
            include_title_if_available=include_title_if_available,
            name=f"query-{i}",
        )

    # get responses and retrieved contexts
    responses: t.List[str] = []
    retrieved_contexts: t.List[t.List[str]] = []
    results = exec.results()

    for r in results:
        responses.append(r.results.generated_answer)
        retrieved_contexts.append(
            _process_search_results(r.results.search_results.as_dict())
        )

    # append the extra information to the dataset
    for i, sample in enumerate(samples):
        sample.response = responses[i]
        sample.retrieved_contexts = retrieved_contexts[i]

    results = ragas_evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=raise_exceptions,
        callbacks=callbacks,
        show_progress=show_progress,
        run_config=run_config or RunConfig(),
        token_usage_parser=token_usage_parser,
    )

    return results
