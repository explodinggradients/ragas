from __future__ import annotations

import logging
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
    knowledge_graph: KnowledgeGraph = field(default_factory=KnowledgeGraph)

    @classmethod
    def from_langchain(
        cls,
        llm: LangchainLLM,
        knowledge_graph: t.Optional[KnowledgeGraph] = None,
    ):
        knowledge_graph = knowledge_graph or KnowledgeGraph()
        return cls(LangchainLLMWrapper(llm), knowledge_graph)

    def generate_with_langchain_docs(
        self,
        documents: t.Sequence[LCDocument],
        test_size: int,
        transforms: t.Optional[Transforms] = None,
        scenarios: t.Optional[QuestionTypes] = None,
        run_config: t.Optional[RunConfig] = None,
        with_debugging_logs=False,
        raise_exceptions: bool = True,
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

        # apply transforms and update the knowledge graph
        apply_transforms(transforms, kg)
        self.knowledge_graph = kg

        return self.generate(
            test_size=test_size,
            scenarios=scenarios,
            run_config=run_config,
            with_debugging_logs=with_debugging_logs,
            raise_exceptions=raise_exceptions,
        )

    def generate(
        self,
        test_size: int,
        scenarios: t.Optional[QuestionTypes] = None,
        run_config: t.Optional[RunConfig] = None,
        with_debugging_logs=False,
        raise_exceptions: bool = True,
    ) -> EvaluationDataset:
        """
        Generate an evaluation dataset based on given scenarios and parameters.

        Parameters
        ----------
        test_size : int
            The number of samples to generate.
        scenarios : Optional[QuestionTypes], optional
            A list of tuples containing scenario generators and their probabilities.
            If None, default scenarios will be used.
        run_config : Optional[RunConfig], optional
            Configuration for running the generation process.
        with_debugging_logs : bool, default False
            If True, enable debug logging for various components.
        raise_exceptions : bool, default True
            If True, raise exceptions during the generation process.

        Returns
        -------
        EvaluationDataset
            A dataset containing the generated evaluation samples.

        Notes
        -----
        This function performs the following steps:
        1. Set up scenarios and debug logging if required.
        2. Generate scenarios using an Executor.
        3. Calculate split values for different scenario types.
        4. Generate samples for each scenario.
        5. Compile the results into an EvaluationDataset.
        """
        scenarios = scenarios or default_scenarios(self.llm)

        if with_debugging_logs:
            # TODO: Edit this before pre-release
            from ragas.utils import patch_logger

            patch_logger("ragas.experimental.testset.generators", logging.DEBUG)
            patch_logger("ragas.experimental.testset.graph", logging.DEBUG)
            patch_logger("ragas.experimental.testset.transforms", logging.DEBUG)

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
            exec.submit(scenario.generate_scenarios, splits[i], self.knowledge_graph)

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
