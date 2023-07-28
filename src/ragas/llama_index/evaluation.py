from __future__ import annotations

import typing as t

from datasets import Dataset

from ragas import evaluate as ragas_evaluate
from ragas.metrics.base import Metric

if t.TYPE_CHECKING:
    from llama_index.indices.query.base import BaseQueryEngine


def evaluate(
    query_engine: BaseQueryEngine,
    metrics: list[Metric],
    questions: list[str],
):
    """
    Run evaluation of llama_index QueryEngine with different metrics

    Parameters
    ----------
    query_engine : BaseQueryEngine
        The QueryEngine that is to be evaluated
    metrics : list[Metric]
        The ragas metrics to use for evaluation.
    questions : list[str]
        List of questions to evaluate on

    Returns
    -------
    Result
        Result object containing the scores of each metric. You can use this do analysis
        later. If the top 3 metrics are provided then it also returns the `ragas_score`
        for the entire pipeline.

    Raises
    ------
    ValueError
        if validation fails because the columns required for the metrics are missing or
        if the columns are of the wrong format.

    Examples
    --------
    Once you have a llama_index QueryEngine created you can use it to evaluate on a list
    of questions.

    ```python
    from ragas.metrics import faithfulness, answer_relevancy, context_relevancy
    from ragas.metrics.critique import harmfulness
    from ragas.llama_index import evaluate

    query_engine = # from llamaindex
    questions: list[str] = [] # from somewhere
    metrics = [faithfulness, answer_relevancy, context_relevancy, harmfulness]

    r = evaluate(query_engine, metrics, questions)

    print(r) # prints the scores of each metric
    r.to_pandas() # returns a pandas dataframe if you want to do further analysis
    ```
    """

    try:
        from llama_index.async_utils import run_async_tasks
    except ImportError:
        raise ImportError(
            "llama_index must be installed to use this function. "
            "Install it with `pip install llama_index`."
        )

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
