import typing as t
from dataclasses import dataclass

from langchain_core.documents import Document as LCDocument

from ragas.embeddings.base import BaseRagasEmbeddings, embedding_factory
from ragas.executor import Executor
from ragas.testsetv3.extractors.base import Extractor
from ragas.testsetv3.extractors.llm_based import LLMbasedExtractor
from ragas.testsetv3.extractors.regex_based import RulebasedExtractor
from ragas.testsetv3.graph import Node, NodeLevel


@dataclass
class DocumentExtractor:
    extractors: t.List[Extractor]
    embedding: t.Optional[BaseRagasEmbeddings] = None

    def __post_init__(self):
        llm_extractors = [
            extractor
            for extractor in self.extractors
            if isinstance(extractor, LLMbasedExtractor)
        ]
        rule_extractor = [
            extractor
            for extractor in self.extractors
            if isinstance(extractor, RulebasedExtractor)
        ]
        self.llm_extractors = (
            LLMbasedExtractor.merge_extractors(*llm_extractors)
            if llm_extractors
            else None
        )
        self.regex_extractors = (
            RulebasedExtractor.merge_extractors(*rule_extractor)
            if rule_extractor
            else None
        )

    async def extract(
        self, inputs: t.Union[t.Sequence[Node], t.Sequence[LCDocument]], **args
    ) -> t.Union[t.Sequence[Node], t.Sequence[LCDocument]]:
        exec = Executor(
            desc="Extraction..",
            keep_progress_bar=True,
            raise_exceptions=True,
            run_config=None,
        )

        if all(isinstance(item, Node) for item in inputs):
            for item in inputs:
                exec.submit(self._extract_from_node, item, **args)
        elif all(isinstance(item, LCDocument) for item in inputs):
            for item in inputs:
                exec.submit(self._extract_from_document, item)
        else:
            raise ValueError("Input must be a list of Nodes or Documents")

        return exec.results()

    async def embed(
        self,
        inputs: t.Union[t.Sequence[Node], t.Sequence[LCDocument]],
        attributes: t.List[str],
        **args,
    ) -> t.Union[t.Sequence[Node], t.Sequence[LCDocument]]:
        exec = Executor(
            desc="Embeddding..",
            keep_progress_bar=True,
            raise_exceptions=True,
            run_config=None,
        )

        if all(isinstance(item, Node) for item in inputs):
            for item in inputs:
                exec.submit(self._embed_from_node, item, attributes, **args)
        elif all(isinstance(item, LCDocument) for item in inputs):
            for item in inputs:
                exec.submit(self._embed_document, item, attributes)
        else:
            raise ValueError("Input must be a list of Nodes or Documents")

        return exec.results()

    async def _extract_from_document(self, doc: LCDocument) -> LCDocument:
        if self.llm_extractors:
            for llm_ext in self.llm_extractors:
                output = await llm_ext.aextract(doc)
                doc.metadata.update(output)
        if self.regex_extractors:
            for reg_ext in self.regex_extractors:
                output = reg_ext.extract(doc)
                doc.metadata.update(output)

        extractive_metadata_keys = []
        for metadata in doc.metadata:
            if isinstance(doc.metadata[metadata], str):
                idx = doc.page_content.find(doc.metadata[metadata])
                if idx != -1:
                    extractive_metadata_keys.append(metadata)
            elif isinstance(doc.metadata[metadata], list) and all(
                isinstance(item, str) for item in doc.metadata[metadata]
            ):
                idx = [doc.page_content.find(item) for item in doc.metadata[metadata]]
                if sum(i != -1 for i in idx) > len(idx) / 2:
                    extractive_metadata_keys.append(metadata)

        doc.metadata["extractive_metadata_keys"] = extractive_metadata_keys

        return doc

    async def _extract_from_node(
        self, node: Node, levels: t.Union[str, t.List[NodeLevel]] = "any"
    ) -> Node:
        if node.level in levels or levels == "any":
            if self.llm_extractors:
                for llm_ext in self.llm_extractors:
                    output = await llm_ext.aextract_from_node(node)
                    node.properties["metadata"].update(output)
            if self.regex_extractors:
                for reg_ext in self.regex_extractors:
                    output = reg_ext.extract_from_node(node)
                    node.properties["metadata"].update(output)

        return node

    async def _embed_document(
        self, doc: LCDocument, attributes: t.List[str]
    ) -> LCDocument:
        self.embedding = (
            self.embedding if self.embedding is not None else embedding_factory()
        )
        for attr in attributes:
            if attr == "page_content":
                item_to_embed = doc.page_content
            elif attr in doc.metadata:
                item_to_embed = doc.metadata.get(attr, "")
            else:
                raise ValueError(f"Attribute {attr} not found in document")

            embedding = await self.embedding.aembed_query(item_to_embed)
            doc.metadata[f"{attr}_embedding"] = embedding

        return doc

    async def _embed_from_node(
        self,
        node: Node,
        attributes: t.List[str],
        levels: t.Union[str, t.List[NodeLevel]] = "any",
    ) -> Node:
        if levels == "any" or (isinstance(levels, list) and node.level in levels):
            self.embedding = (
                self.embedding if self.embedding is not None else embedding_factory()
            )
            for attr in attributes:
                if attr == "page_content":
                    item_to_embed = node.properties["page_content"]
                else:
                    item_to_embed = node.properties["metadata"][attr]

                embedding = await self.embedding.aembed_documents(item_to_embed)
                node.properties["metadata"][f"{attr}_embedding"] = embedding

        return node
