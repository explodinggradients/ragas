import typing as t
from dataclasses import dataclass

from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter


from ragas.testsetv3.graph import Node, Relationship, NodeType

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
                metadata = doc.metadata.get(metadata_key)
                if isinstance(metadata, str):
                    idx = text_chunk.find(metadata)
                    if idx != -1:
                        text_chunk_metadata[metadata_key] = metadata

                elif isinstance(metadata, list):
                    metadata_match_idx = [text_chunk.find(item
                                                          
                                                          ) for item in metadata]
                    metadata_idx = [
                        idx
                        for idx, match_idx in enumerate(metadata_match_idx)
                        if match_idx != -1
                    ]
                    if metadata_idx:
                        text_chunk_metadata[metadata_key] = [
                            metadata[i] for i in metadata_idx
                        ]

            text_chunk_doc = Node(
                label = NodeType.CHUNK,
                properties={"page_content":text_chunk, "metadata":text_chunk_metadata}
            )
            text_chunks_docs.append(text_chunk_doc)

        return text_chunks_docs

    def find_seperator(self, doc: Node):
        for separator in self.separators:
            if doc.properties["page_content"].startswith(separator):
                return separator
            
    def split_documents(self, documents: t.Sequence[LCDocument]):
        nodes = []
        relationships = []
        for doc in documents:
            text_chunks = self.text_splitter.split_text(doc.page_content)
            text_chunks_docs = self.create_docs(text_chunks, doc)
            doc = Node(
                label=NodeType.DOC,
                properties={"page_content":doc.page_content, "metadata":doc.metadata},
                relationships=[])
            nodes.append(doc)
            for chunk in text_chunks_docs:
                seperator = self.find_seperator(chunk)
                relationship = Relationship(
                        source=doc,
                        target=chunk,
                        label="contains",
                        properties={"seperator":seperator or "None"}
                    )
                
                relationships.append(relationship)
                doc.relationships.append(relationship)
                nodes.append(chunk)

            nodes.extend(text_chunks_docs)
            
        return nodes, relationships
                
            
            
