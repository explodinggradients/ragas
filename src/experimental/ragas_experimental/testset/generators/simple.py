import typing as t

import numpy as np
from langchain_core.documents import Document
from ragas_experimental.testset.extractors import (
    DocumentExtractor,
    email_extractor,
    headline_extractor,
    keyphrase_extractor,
    link_extractor,
    summary_extractor,
    title_extractor,
)
from ragas_experimental.testset.generators import QADistribution, TestGenerator
from ragas_experimental.testset.generators.base import TestDataset
from ragas_experimental.testset.graph import Node, NodeLevel
from ragas_experimental.testset.questions import (
    DEFAULT_DISTRIBUTION,
    AbstractQA,
    ComparativeAbstractQA,
    SpecificQA,
)
from ragas_experimental.testset.relationships import (
    Cosine,
    Jaccard,
    RelationshipBuilder,
)
from ragas_experimental.testset.splitters import HeadlineSplitter
from ragas_experimental.testset.utils import rng

from ragas._analytics import TestsetGenerationEvent, track
from ragas.embeddings import embedding_factory
from ragas.executor import Executor
from ragas.llms.base import llm_factory
from ragas.utils import check_if_sum_is_close

abstract_qa = AbstractQA(distribution=DEFAULT_DISTRIBUTION)
comparative_qa = ComparativeAbstractQA(distribution=DEFAULT_DISTRIBUTION)
specific_qa = SpecificQA(distribution=DEFAULT_DISTRIBUTION)

QA_DISTRIBUTION = QADistribution(
    question_types=[abstract_qa, comparative_qa, specific_qa],
    probabilities=[0.6, 0.2, 0.2],
)


class SimpleTestGenerator(TestGenerator):
    def __post_init__(self):
        self.llm = self.llm or llm_factory()
        self.embedding = self.embedding or embedding_factory()

    def _document_exraction(self, docs: t.Sequence[Document]) -> t.Sequence[Document]:
        exec = Executor(
            desc="Document extraction",
            keep_progress_bar=True,
            raise_exceptions=True,
            run_config=None,
        )
        extractors = [
            summary_extractor,
            link_extractor,
            email_extractor,
            keyphrase_extractor,
            title_extractor,
            headline_extractor,
        ]
        doc_extractor = DocumentExtractor(extractors=extractors)
        exec.submit(doc_extractor.extract, docs)
        docs = exec.results()
        return docs

    def generate(
        self,
        docs: t.Sequence[Document],
        test_size: int,
        distribution: QADistribution = QA_DISTRIBUTION,
    ) -> TestDataset:
        if not check_if_sum_is_close(list(distribution.values()), 1.0, 3):
            raise ValueError(
                f"distribution passed do not sum to 1.0 [got {sum(list(distribution.values()))}]. Please check the "
                f"distribution."
            )

        extractors = [
            summary_extractor,
            link_extractor,
            email_extractor,
            keyphrase_extractor,
            title_extractor,
            headline_extractor,
        ]
        doc_extractor = DocumentExtractor(
            extractors=extractors, llm=self.llm, embedding=self.embedding
        )
        docs = doc_extractor.extract(docs)

        splitter = HeadlineSplitter(common_metadata_keys=["source", "title"])
        nodes, relationships = splitter.split_documents(docs, "headlines")

        nodes = doc_extractor.embed(
            nodes,
            ["page_content", "summary"],
            {
                "page_content": [
                    NodeLevel.LEVEL_1,
                    NodeLevel.LEVEL_2,
                    NodeLevel.LEVEL_3,
                ],
                "summary": [NodeLevel.LEVEL_0],
            },
        )
        node_extractor = DocumentExtractor(
            extractors=[keyphrase_extractor], llm=self.llm, embedding=self.embedding
        )
        nodes = node_extractor.extract(
            nodes, [NodeLevel.LEVEL_1, NodeLevel.LEVEL_2, NodeLevel.LEVEL_3]
        )

        jaccard = Jaccard(
            name="jaccard_over_keyphrases",
            attribute1="keyphrases",
            attribute2="keyphrases",
            type="fuzzy",
            threshold=50,
        )
        cosine = Cosine(
            name="summary_similarity",
            attribute1="summary_embedding",
            attribute2="summary_embedding",
        )
        if nodes:
            assert all(
                isinstance(node, Node) for node in nodes
            ), "Nodes must be of type Node"

        nodes, relationships = RelationshipBuilder.form_relations(
            nodes,
            relationships,
            similarity_functions=[jaccard, cosine],
            node_level=NodeLevel.LEVEL_0,
        )

        exec = Executor(
            desc="Generating",
            keep_progress_bar=True,
            raise_exceptions=True,
            run_config=None,
        )

        for qa in distribution.keys():
            qa.nodes = nodes
            qa.relationships = relationships
            if qa.llm is None:
                qa.llm = self.llm
            if qa.embedding is None:
                qa.embedding = self.embedding

        index = 0
        for qa, prob in distribution.items():
            num_samples = int(prob * test_size)
            exec.submit(
                qa.generate_questions, query=None, kwargs=None, num_samples=num_samples
            )
            index += num_samples

        remaining_size = test_size - index
        if remaining_size > 0:
            choices = np.array(distribution.keys())
            prob = np.array(distribution.values())
            random_distribution = rng.choice(choices, p=prob, size=remaining_size)
            for qa in random_distribution:
                exec.submit(
                    qa.generate_questions, query=None, kwargs=None, num_samples=1
                )
        results = exec.results()
        results = TestDataset([result for result in results if result is not None])
        track(
            TestsetGenerationEvent(
                event_type="testset_generation",
                evolution_names=[""],
                evolution_percentages=[0.0],
                num_rows=test_size,
                language="",
                is_experiment=True,
            )
        )
        return results
