from __future__ import annotations

import logging
import typing as t
from uuid import uuid4

from datasets import Dataset

from ragas.embeddings import LlamaIndexEmbeddingsWrapper
from ragas.evaluation import evaluate as ragas_evaluate
from ragas.exceptions import ExceptionInRunner
from ragas.executor import Executor
from ragas.llms import LlamaIndexLLMWrapper
from ragas.run_config import RunConfig

if t.TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import (
        BaseEmbedding as LlamaIndexEmbeddings,
    )
    from llama_index.core.base.llms.base import BaseLLM as LlamaindexLLM

    from ragas.evaluation import Result
    from ragas.metrics.base import Metric


logger = logging.getLogger(__name__)


def evaluate(
    query_engine,
    dataset: Dataset,
    metrics: list[Metric],
    llm: t.Optional[LlamaindexLLM] = None,
    embeddings: t.Optional[LlamaIndexEmbeddings] = None,
    raise_exceptions: bool = False,
    column_map: t.Optional[t.Dict[str, str]] = None,
    run_config: t.Optional[RunConfig] = None,
) -> Result:
    column_map = column_map or {}

    # wrap llms and embeddings
    li_llm = None
    if llm is not None:
        li_llm = LlamaIndexLLMWrapper(llm)
    li_embeddings = None
    if embeddings is not None:
        li_embeddings = LlamaIndexEmbeddingsWrapper(embeddings)

    # validate and transform dataset
    if dataset is None:
        raise ValueError("Provide dataset!")

    exec = Executor(
        desc="Running Query Engine",
        keep_progress_bar=True,
        raise_exceptions=raise_exceptions,
        run_config=run_config,
    )

    # get query
    queries = dataset["question"]
    for i, q in enumerate(queries):
        exec.submit(query_engine.aquery, q, name=f"query-{i}")

    answers: t.List[str] = []
    contexts: t.List[t.List[str]] = []
    try:
        results = exec.results()
        if results == []:
            raise ExceptionInRunner()
    except Exception as e:
        raise e
    else:
        for r in results:
            answers.append(r.response)
            contexts.append([n.node.text for n in r.source_nodes])

    # create HF dataset
    hf_dataset = Dataset.from_dict(
        {
            "question": queries,
            "contexts": contexts,
            "answer": answers,
        }
    )
    if "ground_truth" in dataset.column_names:
        hf_dataset = hf_dataset.add_column(
            name="ground_truth",
            column=dataset["ground_truth"],
            new_fingerprint=str(uuid4()),
        )

    results = ragas_evaluate(
        dataset=hf_dataset,
        metrics=metrics,
        llm=li_llm,
        embeddings=li_embeddings,
        raise_exceptions=raise_exceptions,
    )

    return results
