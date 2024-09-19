import logging
import typing as t
from dataclasses import dataclass, field

from ragas.executor import Executor
from ragas.experimental.testset.extractors.base import BaseExtractor
from ragas.experimental.testset.graph import KnowledgeGraph, Node
from ragas.llms.base import BaseRagasLLM, llm_factory

logger = logging.getLogger(__name__)


@dataclass
class DocumentStore:
    extractors: t.List[BaseExtractor] = field(default_factory=list)
    kg: KnowledgeGraph = field(default_factory=KnowledgeGraph)
    llm: BaseRagasLLM = field(default_factory=llm_factory)

    def __post_init__(self):
        if not self.extractors:
            from ragas.experimental.testset.extractors.embedding import (
                embedding_extractor,
            )
            from ragas.experimental.testset.extractors.llm_based import (
                HeadlinesExtractor,
                KeyphrasesExtractor,
                SummaryExtractor,
                TitleExtractor,
            )
            from ragas.experimental.testset.extractors.regex_based import (
                emails_extractor,
                links_extractor,
                markdown_headings_extractor,
            )

            self.extractors = [
                SummaryExtractor(llm=self.llm),
                KeyphrasesExtractor(llm=self.llm),
                TitleExtractor(llm=self.llm),
                HeadlinesExtractor(llm=self.llm),
                emails_extractor,
                links_extractor,
                markdown_headings_extractor,
                embedding_extractor,
            ]

    def add_extractor(self, extractor: BaseExtractor):
        self.extractors.append(extractor)

    def add_nodes(self, nodes: t.List[Node]):
        # run extrators against the nodes
        exec = Executor(
            desc="Extraction..",
            keep_progress_bar=True,
            raise_exceptions=False,
            run_config=None,
        )

        # Closure function to extract properties and add properties to node.
        async def extract_with_node_id(node: Node, extractor: BaseExtractor):
            key, value = await extractor.extract(node)
            if value is not None:
                try:
                    node.add_property(key, value)
                except ValueError as e:
                    logger.warning(
                        "Error adding property '%s' to node '%s': %s", key, node, e
                    )

        # extract properties from each node and add to knowledge graph
        for node in nodes:
            for extractor in self.extractors:
                exec.submit(extract_with_node_id, node, extractor)
            self.kg._add_node(node)
        exec.results()
