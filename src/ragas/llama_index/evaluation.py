from __future__ import annotations

import typing as t

from datasets import Dataset
from llama_index.async_utils import run_async_tasks
from llama_index.indices.query.base import BaseQueryEngine

from ragas import evaluate as ragas_evaluate
from ragas.metrics.base import Metric


def evaluate(
    query_engine: BaseQueryEngine,
    metrics: list[Metric],
    questions: list[str],
    ground_truths: t.Optional[list[list[str]]] = None,
):
    # TODO: rate limit, error handling, retries
    responses = run_async_tasks([query_engine.aquery(q) for q in questions])

    answers = []
    contexts = []
    for r in responses:
        answers.append(r.response)
        contexts.append([c.node.get_content() for c in r.source_nodes])

    ds = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
    )
    result = ragas_evaluate(ds, metrics)

    return result
