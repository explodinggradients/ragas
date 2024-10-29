from __future__ import annotations

import logging
import datasets
import typing as t
from uuid import uuid4

from ragas.embeddings import LlamaIndexEmbeddingsWrapper
from ragas.evaluation import evaluate as ragas_evaluate
from ragas.exceptions import ExceptionInRunner
from ragas.executor import Executor
from ragas.llms import LlamaIndexLLMWrapper
from ragas.run_config import RunConfig
from ragas.testset.synthesizers.testset_schema import Testset

if t.TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import (
        BaseEmbedding as LlamaIndexEmbeddings,
    )
    from llama_index.core.base.llms.base import BaseLLM as LlamaindexLLM

    from ragas.evaluation import EvaluationResult
    from ragas.metrics.base import Metric


logger = logging.getLogger(__name__)


def evaluate(
    query_engine,
    dataset: Testset,
    metrics: list[Metric],
    llm: t.Optional[LlamaindexLLM] = None,
    embeddings: t.Optional[LlamaIndexEmbeddings] = None,
    raise_exceptions: bool = False,
    column_map: t.Optional[t.Dict[str, str]] = None,
    run_config: t.Optional[RunConfig] = None,
) -> EvaluationResult:
    column_map = column_map or {}

    # wrap llms and embeddings
    li_llm = None
    if llm is not None:
        li_llm = LlamaIndexLLMWrapper(llm, run_config=run_config)
    li_embeddings = None
    if embeddings is not None:
        li_embeddings = LlamaIndexEmbeddingsWrapper(embeddings, run_config=run_config)

    # validate and transform dataset
    if dataset is None:
        raise ValueError("Provide dataset!")

    dataset = dataset.to_hf_dataset()
    
    exec = Executor(
        desc="Running Query Engine",
        keep_progress_bar=True,
        raise_exceptions=raise_exceptions,
        run_config=run_config,
    )

    # get query
    queries = dataset["user_input"]
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
    hf_dataset = datasets.Dataset.from_dict(
        {
            "user_input": queries,
            "retrieved_contexts": contexts,
            "response": answers,
        }
    )
    if "reference" in dataset.column_names:
        hf_dataset = hf_dataset.add_column(
            name="reference",
            column=dataset["reference"],
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
