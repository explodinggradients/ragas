from __future__ import annotations

import typing as t
from collections import defaultdict

from datasets import Dataset
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.schema import RUN_KEY
from langsmith.evaluation import EvaluationResult, RunEvaluator
from langsmith.schemas import Example, Run

from ragas.metrics.base import EvaluationMode, Metric
from ragas.validation import EVALMODE_TO_COLUMNS

if t.TYPE_CHECKING:
    from langchain.callbacks.base import Callbacks


class RagasEvaluatorChain(Chain, RunEvaluator):
    """
    Wrapper around ragas Metrics to use them with langsmith.
    """

    metric: Metric

    def __init__(self, **kwargs: t.Any):
        super().__init__(**kwargs)
        self.metric.init_model()

    @property
    def input_keys(self) -> list[str]:
        keys = ["query", "result"]
        if self.metric.evaluation_mode in [EvaluationMode.qac, EvaluationMode.qc]:
            keys += ["source_documents"]
        if self.metric.evaluation_mode in [EvaluationMode.gc]:
            keys += ["ground_truths"]
        return keys

    @property
    def output_keys(self) -> list[str]:
        return [f"{self.metric.name}_score"]

    def _call(
        self,
        inputs: dict[str, t.Any],
        run_manager: t.Optional[CallbackManagerForChainRun] = None,
    ) -> dict[str, t.Any]:
        """
        Call the evaluation chain.
        """
        self._validate(inputs)
        contexts = []

        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()

        if "source_documents" in inputs:
            for document in inputs["source_documents"]:
                if isinstance(document, dict):
                    contexts.append(document["page_content"])
                else:
                    contexts.append(document.page_content)
        ground_truths = []
        if "ground_truths" in inputs:
            ground_truths = inputs["ground_truths"]

        question = inputs["query"]
        answer = inputs["result"]
        score = self.metric.score_single(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truths": ground_truths,
            },
            callbacks=callbacks,
        )
        return {f"{self.metric.name}_score": score}

    def _validate(
        self,
        input: dict[str, t.Any],
        question_key: str = "query",
        prediction_key: str = "result",
        context_key: str = "source_documents",
    ) -> None:
        ...
        # validate each example
        required_columns = EVALMODE_TO_COLUMNS[self.metric.evaluation_mode]
        if "question" in required_columns and question_key not in input:
            raise ValueError(
                f'"{question_key}" is required in each example'
                f"for the metric[{self.metric.name}] you have chosen."
            )
        if "answer" in required_columns and prediction_key not in input:
            raise ValueError(
                f'"{prediction_key}" is required in each prediction'
                f"for the metric[{self.metric.name}] you have chosen."
            )
        if "contexts" in required_columns and context_key not in input:
            raise ValueError(
                f'"{context_key}" is required in each prediction for the '
                f"metric[{self.metric.name}] you have chosen."
            )
        if "ground_truths" in required_columns and "ground_truths" not in input:
            raise ValueError(
                f'"ground_truths" is required in each prediction for the '
                f"metric[{self.metric.name}] you have chosen."
            )

    def evaluate(
        self,
        examples: t.Sequence[dict],
        predictions: t.Sequence[dict],
        question_key: str = "query",
        prediction_key: str = "result",
        context_key: str = "source_documents",
        ground_truths_key: str = "ground_truths",
        *,
        callbacks: Callbacks = None,
    ) -> list[dict]:
        """Evaluate question answering examples and predictions."""
        dataset_dict = defaultdict(list)

        # validation
        if len(examples) != len(predictions):
            raise ValueError(
                "number of examples and predictions must be same. Got "
                f"len(examples)={len(examples)} and len(predictions)={len(predictions)}"
            )

        for i, example in enumerate(examples):
            self._validate(
                {**example, **predictions[i]}, question_key, prediction_key, context_key
            )
            # transform into Dataset that is supported by ragas
            if self.metric.evaluation_mode in [
                EvaluationMode.qac,
                EvaluationMode.qc,
                EvaluationMode.qa,
            ]:
                dataset_dict["question"].append(example[question_key])

            if self.metric.evaluation_mode in [EvaluationMode.qac, EvaluationMode.qa]:
                dataset_dict["answer"].append(predictions[i][prediction_key])

            if self.metric.evaluation_mode in [
                EvaluationMode.qac,
                EvaluationMode.qc,
                EvaluationMode.gc,
            ]:
                dataset_dict["contexts"].append(
                    [d.page_content for d in predictions[i][context_key]]
                )

            if self.metric.evaluation_mode == EvaluationMode.gc:
                if isinstance(example["ground_truths"], list):
                    dataset_dict["ground_truths"].append(example["ground_truths"])
                else:
                    dataset_dict["ground_truths"].append([example["ground_truths"]])

        dataset = Dataset.from_dict(dataset_dict)

        # evaluate
        dataset_with_scores = self.metric.score(dataset, callbacks=callbacks)
        scores = [
            {f"{self.metric.name}_score": score}
            for score in dataset_with_scores[self.metric.name]
        ]
        return scores

    def evaluate_run(
        self, run: Run, example: t.Optional[Example] = None
    ) -> EvaluationResult:
        """
        Evaluate a langsmith run
        """
        if run.outputs is None:
            raise ValueError("The chain should return results and service_document.")
        if example is None:
            raise ValueError("Examples have to be provided.")
        chain_eval = run.outputs
        chain_eval["query"] = run.inputs["query"]
        if self.metric.evaluation_mode == EvaluationMode.gc:
            if example.outputs is None or "ground_truths" not in example.outputs:
                raise ValueError("expected `ground_truths` in example outputs.")
            chain_eval["ground_truths"] = example.outputs["ground_truths"]
        eval_output = self(chain_eval, include_run_info=True)

        score_name = f"{self.metric.name}_score"
        evaluation_result = EvaluationResult(
            key=f"{self.metric.name}_score", score=eval_output[score_name]
        )
        if RUN_KEY in eval_output:
            evaluation_result.evaluator_info[RUN_KEY] = eval_output[RUN_KEY]
        return evaluation_result
