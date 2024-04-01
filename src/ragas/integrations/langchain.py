from __future__ import annotations

import typing as t

from langchain.chains.base import Chain
from langchain.schema import RUN_KEY
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langsmith.evaluation import EvaluationResult, RunEvaluator
from langsmith.schemas import Example, Run

from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics.base import (
    EvaluationMode,
    Metric,
    MetricWithEmbeddings,
    MetricWithLLM,
    get_required_columns,
)
from ragas.run_config import RunConfig
from ragas.validation import EVALMODE_TO_COLUMNS

if t.TYPE_CHECKING:
    from langchain.callbacks.manager import (
        AsyncCallbackManagerForChainRun,
        CallbackManagerForChainRun,
    )


class EvaluatorChain(Chain, RunEvaluator):
    """
    Wrapper around ragas Metrics to use them with langsmith.
    """

    metric: Metric

    def __init__(self, metric: Metric, **kwargs: t.Any):
        kwargs["metric"] = metric
        super().__init__(**kwargs)
        if "run_config" in kwargs:
            run_config = kwargs["run_config"]
        else:
            run_config = RunConfig()
        if isinstance(self.metric, MetricWithLLM):
            llm = kwargs.get("llm", ChatOpenAI())
            t.cast(MetricWithLLM, self.metric).llm = LangchainLLMWrapper(llm)
        if isinstance(self.metric, MetricWithEmbeddings):
            embeddings = kwargs.get("embeddings", OpenAIEmbeddings())
            t.cast(
                MetricWithEmbeddings, self.metric
            ).embeddings = LangchainEmbeddingsWrapper(embeddings)
        self.metric.init(run_config)

    @property
    def input_keys(self) -> list[str]:
        return get_required_columns(self.metric.evaluation_mode)

    @property
    def output_keys(self) -> list[str]:
        return [self.metric.name]

    def _call(
        self,
        inputs: dict[str, t.Any],
        run_manager: t.Optional[CallbackManagerForChainRun] = None,
    ) -> dict[str, t.Any]:
        """
        Call the evaluation chain.
        """
        self._validate(inputs)
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()

        c = inputs.get("contexts", [""])
        g = inputs.get("ground_truth", "")
        q = inputs.get("question", "")
        a = inputs.get("answer", "")
        score = self.metric.score(
            {
                "question": q,
                "answer": a,
                "contexts": c,
                "ground_truth": g,
            },
            callbacks=callbacks,
        )
        return {self.metric.name: score}

    async def _acall(
        self,
        inputs: t.Dict[str, t.Any],
        run_manager: t.Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> t.Dict[str, t.Any]:
        """
        Call the evaluation chain.
        """
        self._validate(inputs)
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        # TODO: currently AsyncCallbacks are not supported in ragas
        _run_manager.get_child()

        c = inputs.get("contexts", [""])
        g = inputs.get("ground_truth", "")
        q = inputs.get("question", "")
        a = inputs.get("answer", "")
        score = await self.metric.ascore(
            {
                "question": q,
                "answer": a,
                "contexts": c,
                "ground_truth": g,
            },
            callbacks=[],
        )
        return {self.metric.name: score}

    def _validate(
        self,
        input: dict[str, t.Any],
        question_key: str = "question",
        prediction_key: str = "answer",
        context_key: str = "contexts",
    ) -> None:
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
        if "ground_truth" in required_columns and "ground_truth" not in input:
            raise ValueError(
                f'"ground_truth" is required in each prediction for the '
                f"metric[{self.metric.name}] you have chosen."
            )

    @staticmethod
    def _keys_are_present(keys_to_check: list, dict_to_check: dict) -> list[str]:
        return [k for k in keys_to_check if k not in dict_to_check]

    def _validate_langsmith_eval(self, run: Run, example: t.Optional[Example]) -> None:
        if example is None:
            raise ValueError(
                "expected example to be provided. Please check langsmith dataset and ensure valid dataset is uploaded."
            )
        if example.inputs is None:
            raise ValueError(
                "expected example.inputs to be provided. Please check langsmith dataset and ensure valid dataset is uploaded."
            )
        if example.outputs is None:
            raise ValueError(
                "expected example.inputs to be provided. Please check langsmith dataset and ensure valid dataset is uploaded."
            )
        if "question" not in example.inputs or "ground_truth" not in example.outputs:
            raise ValueError(
                "Expected 'question' and 'ground_truth' in example."
                f"Got: {[k for k in example.inputs.keys()]}"
            )
        assert (
            run.outputs is not None
        ), "the current run has no outputs. The chain should output 'answer' and 'contexts' keys."
        output_keys = get_required_columns(
            self.metric.evaluation_mode, ["question", "ground_truth"]
        )
        missing_keys = self._keys_are_present(output_keys, run.outputs)
        if missing_keys:
            raise ValueError(
                "Expected 'answer' and 'contexts' in run.outputs."
                f"Got: {[k for k in run.outputs.keys()]}"
            )

    def evaluate_run(
        self, run: Run, example: t.Optional[Example] = None
    ) -> EvaluationResult:
        """
        Evaluate a langsmith run
        """
        self._validate_langsmith_eval(run, example)

        # this is just to suppress the type checker error
        # actual check and error message is in the _validate_langsmith_eval
        assert run.outputs is not None
        assert example is not None
        assert example.inputs is not None
        assert example.outputs is not None

        chain_eval = run.outputs
        chain_eval["question"] = example.inputs["question"]
        if self.metric.evaluation_mode in [
            EvaluationMode.gc,
            EvaluationMode.ga,
            EvaluationMode.qcg,
            EvaluationMode.qga,
        ]:
            if example.outputs is None or "ground_truth" not in example.outputs:
                raise ValueError("expected `ground_truth` in example outputs.")
            chain_eval["ground_truth"] = example.outputs["ground_truth"]
        eval_output = self.invoke(chain_eval, include_run_info=True)

        evaluation_result = EvaluationResult(
            key=self.metric.name, score=eval_output[self.metric.name]
        )
        if RUN_KEY in eval_output:
            evaluation_result.evaluator_info[RUN_KEY] = eval_output[RUN_KEY]
        return evaluation_result
