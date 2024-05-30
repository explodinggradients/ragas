from __future__ import annotations

import logging
import typing as t
from uuid import uuid4

from datasets import Dataset
from ragas.exceptions import ExceptionInRunner
from ragas.evaluation import evaluate as ragas_evaluate
from ragas.executor import Executor

if t.TYPE_CHECKING:
    from ragas.metrics.base import Metric
    from ragas.llms import LlamaIndexLLMWrapper
    from ragas.embeddings import LlamaIndexEmbeddingsWrapper


logger = logging.getLogger(__name__)


def evaluate(
    query_engine,
    dataset: dict,
    metrics: list[Metric],
    llm: t.Optional[LlamaIndexLLMWrapper] = None,
    embeddings: t.Optional[LlamaIndexEmbeddingsWrapper] = None,
    raise_exceptions: bool = True,
    column_map: t.Optional[t.Dict[str, str]] = None,
):
    column_map = column_map or {}

    # validate and transform dataset
    if dataset is None:
        raise ValueError("Provide dataset!")

    exec = Executor(
        desc="Running Query Engine",
        keep_progress_bar=True,
        raise_exceptions=raise_exceptions,
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
    if "ground_truth" in dataset:
        hf_dataset.add_column(
            name="ground_truth",
            column=dataset["ground_truth"],
            new_fingerprint=str(uuid4()),
        )

    results = ragas_evaluate(
        dataset=hf_dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=raise_exceptions,
    )
