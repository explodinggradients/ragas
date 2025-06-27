"""base class for all type of metrics in ragas"""

__all__ = ["Metric"]

import asyncio
import string
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from tqdm import tqdm

from ..embedding.base import BaseEmbedding
from ..llm import RagasLLM
from ..prompt.base import Prompt
from ..prompt.dynamic_few_shot import DynamicFewShotPrompt
from .result import MetricResult
from ragas_experimental.model.pydantic_model import (
    ExtendedPydanticBaseModel as BaseModel,
)

if t.TYPE_CHECKING:

    from ragas_experimental.project.core import Project
    from ragas_experimental.experiment import Experiment


@dataclass
class Metric(ABC):
    """Base class for all metrics in the LLM evaluation library."""

    name: str
    prompt: t.Optional[t.Union[str, Prompt]] = None
    _response_model: t.Type[BaseModel] = field(init=False)

    def __post_init__(self):
        if isinstance(self.prompt, str):
            self.prompt = Prompt(self.prompt)

    def get_variables(self) -> t.List[str]:
        if isinstance(self.prompt, Prompt):
            fstr = self.prompt.instruction
        else:
            fstr = self.prompt
        vars = [
            field_name
            for _, field_name, _, _ in string.Formatter().parse(fstr)
            if field_name
        ]
        return vars

    def score(self, llm: RagasLLM, **kwargs) -> MetricResult:

        traces = {}
        traces["input"] = kwargs
        prompt_input = self.prompt.format(**kwargs)
        response = llm.generate(prompt_input, response_model=self._response_model)
        traces["output"] = response.model_dump()
        result = MetricResult(**response.model_dump())
        result.traces = traces
        return result

    async def ascore(self, llm: RagasLLM, **kwargs) -> MetricResult:

        traces = {}

        prompt_input = self.prompt.format(**kwargs)
        traces["input"] = prompt_input
        response = await llm.agenerate(
            prompt_input,
            response_model=self._response_model,
        )
        traces["output"] = response.model_dump()
        result = MetricResult(**response.model_dump())  # Fixed missing parentheses
        result.traces = traces
        return result

    def batch_score(
        self,
        llm: RagasLLM,
        inputs: t.List[t.Dict[str, t.Any]],
    ) -> t.List[MetricResult]:
        return [self.score(llm, **input_dict) for input_dict in inputs]

    async def abatch_score(
        self,
        llm: RagasLLM,
        inputs: t.List[t.Dict[str, t.Any]],
    ) -> t.List[MetricResult]:
        async_tasks = []
        for input_dict in inputs:
            # Add reasoning and n to the input parameters
            async_tasks.append(self.ascore(llm, **input_dict))

        # Run all tasks concurrently and return results
        return await asyncio.gather(*async_tasks)

    @abstractmethod
    def get_correlation(self, gold_label, predictions) -> float:
        """
        Calculate the correlation between gold scores and predicted scores.
        This is a placeholder method and should be implemented based on the specific metric.
        """
        pass

    def align(
        self,
        project: "Project",
        experiment_names: t.List[str],
        model: t.Type[BaseModel],
        embedding_model: BaseEmbedding,
        **kwargs: t.Dict[str, t.Any],
    ):
        """
        Args:
            project: The Project instance containing the experiments.
            experiment_names: A list of experiment names to align with the metric.
            model: The Pydantic model used for the experiment data.
            embedding_model: The embedding model used for dynamic few-shot prompting.

        Align the metric with the specified experiments by different optimization methods.
        """

        assert isinstance(self.prompt, Prompt)
        self.prompt = DynamicFewShotPrompt.from_prompt(
            self.prompt, embedding_model, **kwargs
        )
        datasets = []
        for experiment_name in experiment_names:
            experiment_data = project.get_experiment(experiment_name, model)
            experiment_data.load()
            datasets.append(experiment_data)

        total_items = sum([len(dataset) for dataset in datasets])
        input_vars = self.get_variables()
        output_vars = [self.name, f"{self.name}_reason"]
        with tqdm(total=total_items, desc="Processing examples") as pbar:
            for dataset in datasets:
                for row in dataset:
                    inputs = {
                        var: getattr(row, var)
                        for var in input_vars
                        if hasattr(row, var)
                    }
                    output = {
                        var: getattr(row, var)
                        for var in output_vars
                        if hasattr(row, var)
                    }
                    if output:
                        self.prompt.add_example(inputs, output)
                    pbar.update(1)

    def validate_alignment(
        self,
        llm: RagasLLM,
        gold_experiment: "Experiment",
        mapping: t.Dict[str, str] = {},
    ):
        """
        Args:
            llm: The LLM instance to use for scoring.
            gold_experiment: An Experiment instance containing the gold standard scores.
            mapping: A dictionary mapping variable names expected by metrics to their corresponding names in the gold experiment.

        Validate the alignment of the metric by comparing the scores against a gold standard experiment.
        This method computes the Cohen's Kappa score and agreement rate between the gold standard scores and
        the predicted scores from the metric.
        """

        gold_experiment.load()
        gold_scores = [getattr(row, self.name) for row in gold_experiment]
        pred_scores = []
        for row in tqdm(gold_experiment):
            values = {
                v: (
                    getattr(row, v)
                    if v not in mapping
                    else getattr(row, mapping.get(v, v))
                )
                for v in self.get_variables()
            }
            score = self.score(llm=llm, **values)
            pred_scores.append(score.result)

        df = gold_experiment.to_pandas()
        df[f"{self.name}_pred"] = pred_scores
        correlation = self.get_correlation(gold_scores, pred_scores)
        agreement_rate = sum(x == y for x, y in zip(gold_scores, pred_scores)) / len(
            gold_scores
        )
        return {
            "correlation": correlation,
            "agreement_rate": agreement_rate,
            "df": df,
        }
