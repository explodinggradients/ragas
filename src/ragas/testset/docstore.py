import heapq
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt
from langchain.text_splitter import TextSplitter
from langchain_core.documents import Document as LCDocument

Embedding = t.Union[t.List[float], npt.NDArray[np.float64]]


class Document(LCDocument):
    doc_id: str
    filename: t.Optional[str] = None
    embedding: t.Optional[Embedding] = None


class DocumentStore(ABC):
    def __init__(self):
        self.documents = {}

    @abstractmethod
    def add_documents(self, docs: t.List[Document]):
        ...

    @abstractmethod
    def get_document(self, doc_id: int) -> Document:
        ...

    @abstractmethod
    def get_similar(self, doc: Document) -> t.List[Document]:
        ...

    @abstractmethod
    def get_adjascent(self, doc: Document, direction: str = "next") -> t.List[Document]:
        ...


class SimilarityMode(str, Enum):
    """Modes for similarity/distance."""

    DEFAULT = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


def similarity(
    embedding1: Embedding,
    embedding2: Embedding,
    mode: SimilarityMode = SimilarityMode.DEFAULT,
) -> float:
    """Get embedding similarity."""
    if mode == SimilarityMode.EUCLIDEAN:
        # Using -euclidean distance as similarity to achieve same ranking order
        return -float(np.linalg.norm(np.array(embedding1) - np.array(embedding2)))
    elif mode == SimilarityMode.DOT_PRODUCT:
        return np.dot(embedding1, embedding2)
    else:
        product = np.dot(embedding1, embedding2)
        norm = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        return product / norm


default_similarity_fns = similarity


def get_top_k_embeddings(
    query_embedding: t.List[float],
    embeddings: t.List[t.List[float]],
    similarity_fn: t.Optional[t.Callable[..., float]] = None,
    similarity_top_k: t.Optional[int] = None,
    embedding_ids: t.Optional[t.List] = None,
    similarity_cutoff: t.Optional[float] = None,
) -> t.Tuple[t.List[float], t.List]:
    """Get top nodes by similarity to the query."""
    if embedding_ids is None:
        embedding_ids = list(range(len(embeddings)))

    similarity_fn = similarity_fn or default_similarity_fns

    embeddings_np = np.array(embeddings)
    query_embedding_np = np.array(query_embedding)

    similarity_heap: t.List[t.Tuple[float, t.Any]] = []
    for i, emb in enumerate(embeddings_np):
        similarity = similarity_fn(query_embedding_np, emb)
        if similarity_cutoff is None or similarity > similarity_cutoff:
            heapq.heappush(similarity_heap, (similarity, embedding_ids[i]))
            if similarity_top_k and len(similarity_heap) > similarity_top_k:
                heapq.heappop(similarity_heap)
    result_tups = sorted(similarity_heap, key=lambda x: x[0], reverse=True)

    result_similarities = [s for s, _ in result_tups]
    result_ids = [n for _, n in result_tups]

    return result_similarities, result_ids


class InMemoryDocumentStore:
    def __init__(self, splitter):
        super().__init__()
        self.splitter = splitter
        self.embeddings = {}
        self.documents = []

    def add_documents(self, docs: t.List[Document]):
        return None

    def get_document(self, doc_id: int) -> Document:
        return self.documents[doc_id]

    def get_similar(self, doc: Document, threshold: float = 0.7) -> t.List[Document]:
        return []

    def get_adjascent(
        self, doc: Document, direction: str = "next"
    ) -> t.Optional[t.List[Document]]:
        index = self.documents.index(doc)
        if direction == "next":
            if len(self.documents) > index + 1:
                next_doc = self.documents[index + 1]
                if next_doc.filename == doc.filename:
                    return next_doc
                else:
                    return None
            else:
                return None
        if direction == "prev":
            if index > 0:
                prev_doc = self.documents[index - 1]
                if prev_doc.filename == doc.filename:
                    return prev_doc
                else:
                    return None
            else:
                return None
