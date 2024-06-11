import re
import typing as t
from dataclasses import dataclass

from langchain_core.documents import Document as LCDocument

from ragas.testsetv3.graph import Node, NodeType, Relationship
from ragas.testsetv3.utils import merge_dicts


class KeywordSplitter:
    """
    split text by using given keywords as boundaries. Any chunk that goes beyond chunk_size will also be split to fit max_chunk_size.
    """

    def __init__(self, chunk_size: int = 4000, min_chunk_size: int = 400):
        self._length_function = len
        self._chunk_size = chunk_size
        self._min_chunk_size = min_chunk_size

    def _split_text_by_keywords(self, text: str, keywords: t.List[str]):
        pattern = re.compile("|".join(map(re.escape, keywords)))
        chunks = pattern.split(text)
        matches = pattern.findall(text)
        matches.insert(0, "")

        chunks = [f"{match} {chunk}" for chunk, match in zip(chunks, matches)]
        return chunks, matches

    def _split_text(self, text: str, keywords: t.List[str]) -> t.List[LCDocument]:
        def split_large_chunk(chunk, max_size):
            return [chunk[i : i + max_size] for i in range(0, len(chunk), max_size)]

        chunks, matches = self._split_text_by_keywords(text, keywords)
        # Adjust chunks to ensure no chunk exceeds max_chunk_size and combine small chunks
        adjusted_chunks = []
        adjusted_matches = []

        current_chunk = ""
        current_match = ""

        for match, chunk in zip(matches, chunks):
            if self._length_function(chunk) > self._chunk_size:
                if current_chunk:
                    adjusted_chunks.append(current_chunk)
                    adjusted_matches.append(current_match)
                    current_match = ""
                    current_chunk = ""
                split_chunks = split_large_chunk(chunk, self._chunk_size)
                adjusted_chunks.extend(split_chunks)
                adjusted_matches.extend([match] * len(split_chunks))
            elif (
                self._length_function(current_chunk) + self._length_function(chunk)
                < self._min_chunk_size
            ):
                current_chunk += chunk
                current_match = f"{current_match}-{match}"
            else:
                if current_chunk:
                    adjusted_chunks.append(current_chunk)
                    adjusted_matches.append(current_match)
                current_chunk = chunk
                current_match = match

        if current_chunk:
            adjusted_chunks.append(current_chunk)

        documents = []

        for match, chunk in zip(adjusted_matches, adjusted_chunks):
            metadata = {"keyword": match}
            documents.append(LCDocument(page_content=chunk, metadata=metadata))
        return documents

    def split_text(self, text: str, keywords: t.List[str]) -> t.List[LCDocument]:
        return self._split_text(text, keywords)


@dataclass
class DocumentSplitter:
    """
    reponsible for splitting the document into multiple parts and reassinging extracted metadata to the parts
    """

    def create_nodes(
        self, document: LCDocument, doc_chunks: t.Sequence[LCDocument]
    ) -> t.List[Node]:
        chunk_nodes = []
        extractive_metadata_keys = document.metadata.get("extractive_metadata_keys", [])
        for idx, doc in enumerate(doc_chunks):
            page_content = doc.page_content
            text_chunk_metadata = {}
            for metadata_key in extractive_metadata_keys:
                metadata = document.metadata.get(metadata_key)
                if isinstance(metadata, str):
                    idx = page_content.find(metadata)
                    if idx != -1:
                        text_chunk_metadata[metadata_key] = metadata

                elif isinstance(metadata, list):
                    metadata_match_idx = [page_content.find(item) for item in metadata]
                    metadata_idx = [
                        idx
                        for idx, match_idx in enumerate(metadata_match_idx)
                        if match_idx != -1
                    ]
                    if metadata_idx:
                        text_chunk_metadata[metadata_key] = [
                            metadata[i] for i in metadata_idx
                        ]
            text_chunk_metadata = merge_dicts(doc.metadata, text_chunk_metadata)
            text_chunk_doc = Node(
                label=NodeType.CHUNK,
                properties={
                    "page_content": page_content,
                    "metadata": text_chunk_metadata,
                },
            )
            chunk_nodes.append(text_chunk_doc)

        return chunk_nodes

    def from_keywords(self, documents: t.Sequence[LCDocument], keyword_attribute: str):
        text_splitter = KeywordSplitter()
        nodes = []
        relationships = []
        for doc in documents:
            keywords = doc.metadata.get(keyword_attribute, [])
            chunk_docs = text_splitter.split_text(doc.page_content, keywords)
            chunk_nodes = self.create_nodes(doc, chunk_docs)
            doc = Node(
                label=NodeType.DOC,
                properties={"page_content": doc.page_content, "metadata": doc.metadata},
                relationships=[],
            )

            nodes.append(doc)
            for chunk in chunk_nodes:
                relationship = Relationship(
                    source=doc,
                    target=chunk,
                    label="contains",
                    properties={
                        "seperator": chunk.properties["metadata"].get("keyword")
                    },
                )

                relationships.append(relationship)
                doc.relationships.append(relationship)
                nodes.append(chunk)

        return nodes, relationships
