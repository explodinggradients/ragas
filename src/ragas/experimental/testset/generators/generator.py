from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

from ragas.dataset_schema import EvaluationDataset
from ragas.executor import Executor
from ragas.experimental.testset.generators import default_scenarios
from ragas.experimental.testset.generators.utils import calculate_split_values
from ragas.experimental.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.experimental.testset.transforms import (
    Transforms,
    apply_transforms,
    default_transforms,
)
from ragas.llms import BaseRagasLLM, LangchainLLMWrapper
from ragas.run_config import RunConfig

if t.TYPE_CHECKING:
    from langchain_core.documents import Document as LCDocument
    from langchain_core.language_models import BaseLanguageModel as LangchainLLM

    from ragas.experimental.testset.generators import QuestionTypes
    from ragas.experimental.testset.generators.base import BasicScenario


@dataclass
class TestsetGenerator:
    llm: BaseRagasLLM
    docstore: KnowledgeGraph = field(default_factory=KnowledgeGraph)

    @classmethod
    def from_langchain(
        cls,
        llm: LangchainLLM,
        docstore: t.Optional[KnowledgeGraph] = None,
    ):
        docstore = docstore or KnowledgeGraph()
        return cls(LangchainLLMWrapper(llm), docstore)

    def generate_with_langchain_docs(
        self,
        documents: t.Sequence[LCDocument],
        test_size: int,
        transforms: t.Optional[Transforms] = None,
        scenarios: t.Optional[QuestionTypes] = None,
        with_debugging_logs=False,
        raise_exceptions: bool = True,
        run_config: t.Optional[RunConfig] = None,
    ) -> EvaluationDataset:
        transforms = transforms or default_transforms()

        # convert the documents to Ragas nodes
        nodes = []
        for doc in documents:
            node = Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata,
                },
            )
            nodes.append(node)

        kg = KnowledgeGraph(nodes=nodes)

        # apply transforms and update the docstore
        apply_transforms(transforms, kg)
        self.docstore = kg

        return self.generate(
            test_size,
            scenarios,
            with_debugging_logs,
            raise_exceptions,
            run_config,
        )

    def generate(
        self,
        test_size: int,
        scenarios: t.Optional[QuestionTypes] = None,
        with_debugging_logs=False,
        raise_exceptions: bool = True,
        run_config: t.Optional[RunConfig] = None,
    ) -> EvaluationDataset:
        scenarios = scenarios or default_scenarios(self.llm)
        # generate scenarios
        exec = Executor(
            "Generating Scenarios",
            raise_exceptions=raise_exceptions,
            run_config=run_config,
            keep_progress_bar=False,
        )
        # generate samples
        splits, split_values = calculate_split_values(
            [prob for _, prob in scenarios], test_size
        )
        for i, (scenario, _) in enumerate(scenarios):
            exec.submit(scenario.generate_scenarios, splits[i], self.docstore)

        scenario_sample_list: t.List[t.List[BasicScenario]] = exec.results()

        exec = Executor(
            "Generating Samples",
            raise_exceptions=raise_exceptions,
            run_config=run_config,
            keep_progress_bar=True,
        )
        for i, (scenario, _) in enumerate(scenarios):
            for sample in scenario_sample_list[i]:
                exec.submit(scenario.generate_sample, sample)

        eval_samples = exec.results()
        return EvaluationDataset(samples=eval_samples)
