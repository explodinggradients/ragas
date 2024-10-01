from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

from ragas.executor import Executor
from ragas.experimental.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.experimental.testset.synthesizers import default_query_distribution
from ragas.experimental.testset.synthesizers.testset_schema import Testset, TestsetSample
from ragas.experimental.testset.synthesizers.utils import calculate_split_values
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

    from ragas.experimental.testset.synthesizers import QueryDistribution
    from ragas.experimental.testset.synthesizers.base import BaseScenario


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
        query_distribution: t.Optional[QueryDistribution] = None,
        run_config: t.Optional[RunConfig] = None,
        with_debugging_logs=False,
        raise_exceptions: bool = True,
    ) -> Testset:
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
        apply_transforms(kg, transforms)
        self.knowledge_graph = kg

        return self.generate(
            test_size=test_size,
            query_distribution=query_distribution,
            run_config=run_config,
            with_debugging_logs=with_debugging_logs,
            raise_exceptions=raise_exceptions,
        )

    def generate(
        self,
        test_size: int,
        query_distribution: t.Optional[QueryDistribution] = None,
        run_config: t.Optional[RunConfig] = None,
        with_debugging_logs=False,
        raise_exceptions: bool = True,
    ) -> Testset:
        """
        Generate an evaluation dataset based on given scenarios and parameters.

        Parameters
        ----------
        test_size : int
            The number of samples to generate.
        query_distribution : Optional[QueryDistribution], optional
            A list of tuples containing scenario simulators and their probabilities.
            If None, default simulators will be used.
        run_config : Optional[RunConfig], optional
            Configuration for running the generation process.
        with_debugging_logs : bool, default False
            If True, enable debug logging for various components.
        raise_exceptions : bool, default True
            If True, raise exceptions during the generation process.

        Returns
        -------
        Testset
            A dataset containing the generated TestsetSamples.

        Notes
        -----
        This function performs the following steps:
        1. Set up scenarios and debug logging if required.
        2. Generate scenarios using an Executor.
        3. Calculate split values for different scenario types.
        4. Generate samples for each scenario.
        5. Compile the results into an EvaluationDataset.
        """
        query_distribution = (
            query_distribution or default_query_distribution(self.llm)
        )

        if with_debugging_logs:
            # TODO: Edit this before pre-release
            from ragas.utils import patch_logger

            patch_logger("ragas.experimental.testset.synthesizers", logging.DEBUG)
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
        splits, _ = calculate_split_values(
            [prob for _, prob in query_distribution], test_size
        )
        for i, (scenario, _) in enumerate(query_distribution):
            exec.submit(scenario.generate_scenarios, splits[i], self.knowledge_graph)

        scenario_sample_list: t.List[t.List[BaseScenario]] = exec.results()

        exec = Executor(
            "Generating Samples",
            raise_exceptions=raise_exceptions,
            run_config=run_config,
            keep_progress_bar=True,
        )
        additional_testset_info: t.List[t.Dict] = []
        for i, (synthesizer, _) in enumerate(query_distribution):
            for sample in scenario_sample_list[i]:
                exec.submit(synthesizer.generate_sample, sample)
                # fill out the additional info for the TestsetSample
                additional_testset_info.append(
                    {
                        "synthesizer_name": synthesizer.name,
                    }
                )

        eval_samples = exec.results()

        # build the testset
        testsets = []
        for sample, additional_info in zip(eval_samples, additional_testset_info):
            testsets.append(TestsetSample(eval_sample=sample, **additional_info))
        return Testset(samples=testsets)
