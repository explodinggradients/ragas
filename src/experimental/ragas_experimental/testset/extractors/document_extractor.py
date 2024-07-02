import typing as t

from langchain_core.documents import Document as LCDocument

from ragas.embeddings.base import BaseRagasEmbeddings, embedding_factory
from ragas.executor import Executor
from ragas.llms.base import BaseRagasLLM, llm_factory
from ragas_experimental.testset.extractors.base import Extractor
from ragas_experimental.testset.extractors.llm_based import LLMbasedExtractor
from ragas_experimental.testset.extractors.regex_based import RulebasedExtractor
from ragas_experimental.testset.graph import Node, NodeLevel


class DocumentExtractor:
    def __init__(
        self,
        extractors: t.List[Extractor],
        llm: t.Optional[BaseRagasLLM] = None,
        embedding: t.Optional[BaseRagasEmbeddings] = None,
    ):
        self.extractors = extractors
        self.llm = llm or llm_factory()
        self.embedding = embedding or embedding_factory()

    @property
    def extractors(self):
        return self._extractors

    @extractors.setter
    def extractors(self, extractors):
        self._extractors = []
        llm_extractors = [
            extractor
            for extractor in extractors
            if isinstance(extractor, LLMbasedExtractor)
        ]
        rule_extractor = [
            extractor
            for extractor in extractors
            if isinstance(extractor, RulebasedExtractor)
        ]
        if llm_extractors:
            self._extractors.extend(LLMbasedExtractor.merge_extractors(*llm_extractors))
        if rule_extractor:
            self._extractors.extend(
                RulebasedExtractor.merge_extractors(*rule_extractor)
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
        for extractor in self._extractors:
            if isinstance(extractor, LLMbasedExtractor):
                output = await extractor.aextract(doc)
                doc.metadata.update(output)
            elif isinstance(extractor, RulebasedExtractor):
                output = extractor.extract(doc)
                doc.metadata.update(output)
            else:
                raise ValueError("Extractor not supported")

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
        if levels == "any" or (isinstance(levels, list) and node.level in levels):
            for extractor in self._extractors:
                if isinstance(extractor, LLMbasedExtractor):
                    output = await extractor.aextract(node)
                    node.properties["metadata"].update(output)
                elif isinstance(extractor, RulebasedExtractor):
                    output = extractor.extract(node)
                    node.properties["metadata"].update(output)
                else:
                    raise ValueError("Extractor not supported")

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
