"""base class for all type of metrics in ragas"""

__all__ = ["Metric"]

import asyncio
import string
import typing as t
from abc import ABC
from dataclasses import dataclass, field

from pydantic import BaseModel
from tqdm import tqdm

from ..embedding.base import BaseEmbedding
from ..llm import RagasLLM
from ..model.notion_model import NotionModel
from ..prompt.base import Prompt
from ..prompt.dynamic_few_shot import DynamicFewShotPrompt
from .result import MetricResult

if t.TYPE_CHECKING:
    from ragas_experimental.project.core import Project


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

    def train(
        self,
        project: "Project",
        experiment_names: t.List[str],
        model: NotionModel,
        embedding_model: BaseEmbedding,
        method: t.Dict[str, t.Any],
    ):

        assert isinstance(self.prompt, Prompt)
        self.prompt = DynamicFewShotPrompt.from_prompt(self.prompt, embedding_model)
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
