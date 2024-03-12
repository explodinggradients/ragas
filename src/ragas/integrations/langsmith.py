import typing as t

from langchain.smith import RunEvalConfig
from langsmith import Client
from langsmith.utils import LangSmithNotFoundError


from ragas.integrations.langchain import EvaluatorChain


def evaluate(
    dataset_name: str,
    llm_or_chain_factory: t.Any,
    run_name: str = "",
    metrics: t.Optional[list] = None,
    verbose: bool = False,
) -> t.Dict[str, t.Any]:
    # get sensible run name
    if not run_name:
        run_name = llm_or_chain_factory.get_name()
    # init client and validate dataset
    client = Client()
    try:
        _ = client.read_dataset(dataset_name=dataset_name)
    except LangSmithNotFoundError:
        raise ValueError(
            f"Dataset {dataset_name} not found in langsmith, make sure it exists in langsmith"
        )

    # make config
    if metrics is None:
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        metrics = [answer_relevancy, context_precision, faithfulness, context_recall]

    metrics = [EvaluatorChain(m) for m in metrics]
    eval_config = RunEvalConfig(
        custom_evaluators=metrics,
    )

    # run evaluation with langsmith
    run = client.run_on_dataset(
        dataset_name=dataset_name,
        llm_or_chain_factory=llm_or_chain_factory,
        evaluation=eval_config,
        verbose=verbose,
        # Any experiment metadata can be specified here
    )

    return run
