from __future__ import annotations

import typing as t

from langchain.chains.base import Chain
from langchain.schema import RUN_KEY
from langsmith.evaluation import EvaluationResult, RunEvaluator
from langsmith.schemas import Example, Run

from ragas.metrics.base import Metric

if t.TYPE_CHECKING:
    from langchain.callbacks.base import Callbacks


class RagasEvalutorChain(Chain, RunEvaluator):
    metric: Metric

    @property
    def input_keys(self):
        return ["query", "result", "source_documents"]

    @property
    def output_keys(self):
        return [f"{self.metric.name}_score"]

    def _call(
        self,
        inputs,
        run_manager=None,
    ):
        """Call the evaluation chain."""
        answer = inputs["result"]
        question = inputs["query"]
        contexts = []
        if "source_documents" in inputs:
            contexts = [d.page_content for d in inputs["source_documents"]]

        score = self.metric.score_single(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
            }
        )
        print(score)
        return {f"{self.metric.name}_score": score}

    def evaluate(
        self,
        examples: t.Sequence[dict],
        predictions: t.Optional[t.Sequence[dict]] = None,
        question_key: str = "query",
        answer_key: str = "answer",
        prediction_key: str = "result",
        *,
        callbacks: Callbacks = None,
    ) -> list[dict]:
        """Evaluate question answering examples and predictions."""

    def evaluate_run(
        self, run: Run, example: t.Optional[Example] = None
    ) -> EvaluationResult:
        if run.outputs is None:
            raise ValueError("Run outputs cannot be None")
        eval_input = run.outputs["query"] = run.inputs["query"]
        eval_output = self(run.outputs, include_run_info=True)

        score_name = f"{self.metric.name}_score"
        evaluation_result = EvaluationResult(
            key=self.metric.name, score=eval_output[score_name]
        )
        if RUN_KEY in eval_output:
            evaluation_result.evaluator_info[RUN_KEY] = eval_output[RUN_KEY]
        return evaluation_result
