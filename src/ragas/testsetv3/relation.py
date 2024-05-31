import typing as t
from dataclasses import dataclass

from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class DocumentSplitter:
    """
    reponsible for splitting the document into multiple parts and reassinging extracted metadata to the parts
    """

    separators: t.List[str]

    def __post_init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(separators=self.separators)

    def create_docs(self, text_chunks: t.Sequence[str], doc: LCDocument):
        text_chunks_docs = []
        extractive_metadata_keys = doc.metadata.get("extractive_metadata_keys", [])
        for idx, text_chunk in enumerate(text_chunks):
            text_chunk_metadata = {}
            for metadata_key in extractive_metadata_keys:
                metadata = doc.metadata[metadata_key]
                if isinstance(metadata, str):
                    idx = text_chunk.find(metadata)
                    if idx != -1:
                        text_chunk_metadata[metadata_key] = metadata

                elif isinstance(metadata, list):
                    metadata_match_idx = [text_chunk.find(item) for item in metadata]
                    metadata_idx = [
                        idx
                        for idx, match_idx in enumerate(metadata_match_idx)
                        if match_idx != -1
                    ]
                    if metadata_idx:
                        text_chunk_metadata[metadata_key] = [
                            metadata[i] for i in metadata_idx
                        ]

            text_chunk_doc = LCDocument(
                page_content=text_chunk, metadata=text_chunk_metadata
            )
            text_chunks_docs.append(text_chunk_doc)

        return text_chunks_docs

    def split_documents(self, documents: t.Sequence[LCDocument]):
        for doc in documents:
            text_chunks = self.text_splitter.split_text(doc.page_content)
            text_chunks_docs = self.create_docs(text_chunks, doc)
            return text_chunks_docs


class UnstructuredRelationExtractor:
    def __init__(self):
        pass

    def __call__(
        self,
    ):
        pass
