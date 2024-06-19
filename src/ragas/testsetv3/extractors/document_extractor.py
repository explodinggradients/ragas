import typing as t
from dataclasses import dataclass

from langchain_core.documents import Document as LCDocument

from ragas.embeddings.base import BaseRagasEmbeddings, embedding_factory
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

    async def extract_from_documents(self, documents: t.Sequence[LCDocument]):
        for doc in documents:
            if self.llm_extractors:
                output = await self.llm_extractors.extract(doc.page_content)
                doc.metadata.update(output)
            if self.regex_extractors:
                output = self.regex_extractors.extract(doc.page_content)
                doc.metadata.update(output)

        doc = documents[0]
        extractive_metadata_keys = []
        for metadata in doc.metadata:
            if isinstance(doc.metadata[metadata], str):
                idx = doc.page_content.find(doc.metadata[metadata])
                if idx != -1:
                    extractive_metadata_keys.append(metadata)
            elif isinstance(doc.metadata[metadata], list):
                idx = [doc.page_content.find(item) for item in doc.metadata[metadata]]
                if sum(i != -1 for i in idx) > len(idx) / 2:
                    extractive_metadata_keys.append(metadata)

        for doc in documents:
            doc.metadata["extractive_metadata_keys"] = extractive_metadata_keys

        return documents

    async def extract_from_nodes(self, nodes: t.List[Node], levels: t.List[NodeLevel]):
        for node in nodes:
            if node.level in levels:
                if self.llm_extractors:
                    output = await self.llm_extractors.extract(
                        node.properties["page_content"]
                    )
                    node.properties["metadata"].update(output)
                if self.regex_extractors:
                    output = self.regex_extractors.extract(
                        node.properties["page_content"]
                    )
                    node.properties["metadata"].update(output)

        return nodes

    async def embed_from_documents(
        self, documents: t.Sequence[LCDocument], attributes: t.List[str]
    ):
        self.embedding = (
            self.embedding if self.embedding is not None else embedding_factory()
        )
        for attr in attributes:
            if attr == "page_content":
                items_to_embed = [doc.page_content for doc in documents]
            elif attr in documents[0].metadata:
                items_to_embed = [doc.metadata.get(attr, "") for doc in documents]
            else:
                raise ValueError(f"Attribute {attr} not found in document")

            embeddings_list = await self.embedding.aembed_documents(items_to_embed)
            assert len(embeddings_list) == len(
                items_to_embed
            ), "Embeddings and document must be of equal length"
            for doc, embedding in zip(documents, embeddings_list):
                doc.metadata[f"{attr}_embedding"] = embedding

        return documents

    async def embed_from_nodes(
        self, nodes: t.List[Node], attributes: t.List[str], levels: t.List[NodeLevel]
    ):
        self.embedding = (
            self.embedding if self.embedding is not None else embedding_factory()
        )
        nodes_ = [node for node in nodes if node.level in levels]
        for attr in attributes:
            if attr == "page_content":
                items_to_embed = [node.properties["page_content"] for node in nodes_]
            else:
                items_to_embed = [node.properties["metadata"][attr] for node in nodes_]

            embeddings_list = await self.embedding.aembed_documents(items_to_embed)
            assert len(embeddings_list) == len(
                items_to_embed
            ), "Embeddings and document must be of equal length"
            for node, embedding in zip(nodes_, embeddings_list):
                node.properties["metadata"][f"{attr}_embedding"] = embedding

        return nodes
