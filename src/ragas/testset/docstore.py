import heapq
import typing as t
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import numpy.typing as npt
from langchain.text_splitter import TextSplitter
from langchain_core.documents import Document as LCDocument
from pydantic import Field

from ragas.async_utils import run_async_tasks
from ragas.embeddings.base import BaseRagasEmbeddings, embedding_factory

Embedding = t.Union[t.List[float], npt.NDArray[np.float64]]


class Document(LCDocument):
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: t.Optional[str] = None
    embedding: t.Optional[t.List[float]] = Field(default=None, repr=False)


class DocumentStore(ABC):
    def __init__(self):
        self.documents = {}

    @abstractmethod
    def add(self, doc: t.Union[Document, t.Sequence[Document]], show_progress=True):
        ...

    @abstractmethod
    def get(self, doc_id: str) -> Document:
        ...

    @abstractmethod
    def get_similar(
        self, doc: Document, threshold: float = 0.7, top_k: int = 3
    ) -> t.List[Document]:
        ...

    @abstractmethod
    def get_adjascent(
        self, doc: Document, direction: str = "next"
    ) -> t.Optional[Document]:
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
    query_embedding: Embedding,
    embeddings: t.List[Embedding],
    similarity_fn: t.Optional[t.Callable[..., float]] = None,
    similarity_top_k: t.Optional[int] = None,
    embedding_ids: t.Optional[t.List] = None,
    similarity_cutoff: t.Optional[float] = None,
) -> t.Tuple[t.List[float], t.List]:
    """
    Get top nodes by similarity to the query.
    returns the scores and the embedding_ids of the nodes
    """
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


@dataclass
class InMemoryDocumentStore(DocumentStore):
    splitter: TextSplitter
    embeddings: BaseRagasEmbeddings = field(
        default_factory=embedding_factory, repr=False
    )
    documents_list: t.List[Document] = field(default_factory=list)
    embeddings_list: t.List[Embedding] = field(default_factory=list)
    documents_map: t.Dict[str, Document] = field(default_factory=dict)

    def _add_documents_batch(self, docs: t.Sequence[Document], show_progress=True):
        """
        Add documents in batch mode.
        """
        # NOTE: Adds everything in async mode for now.
        embed_tasks = []
        docs_to_embed = []
        for doc in docs:
            if doc.embedding is None:
                embed_tasks.append(self.embeddings.aembed_query(doc.page_content))
                docs_to_embed.append(doc)
            else:
                self.documents_list.append(doc)
                self.documents_map[doc.doc_id] = doc
                self.embeddings_list.append(doc.embedding)

        embeddings = run_async_tasks(embed_tasks, show_progress=show_progress)
        for doc, embedding in zip(docs_to_embed, embeddings):
            doc.embedding = embedding
            self.documents_list.append(doc)
            self.documents_map[doc.doc_id] = doc
            self.embeddings_list.append(doc.embedding)

    def add(self, doc: t.Union[Document, t.Sequence[Document]], show_progress=True):
        if isinstance(doc, list) or isinstance(doc, tuple):
            self._add_documents_batch(doc)
        elif isinstance(doc, Document):
            self.documents_list.append(doc)
            self.documents_map[doc.doc_id] = doc
            if doc.embedding is None:
                doc.embedding = self.embeddings.embed_query(doc.page_content)
            self.embeddings_list.append(doc.embedding)
        else:
            raise ValueError("add() method only supports Document or List[Document]")

    def get(self, doc_id: str) -> Document:
        return self.documents_map[doc_id]

    def get_similar(
        self, doc: Document, threshold: float = 0.7, top_k: int = 3
    ) -> t.List[Document]:
        if doc.embedding is None:
            raise ValueError("Document has no embedding.")
        scores, doc_ids = get_top_k_embeddings(
            query_embedding=doc.embedding,
            embeddings=self.embeddings_list,
            similarity_fn=similarity,
            similarity_cutoff=threshold,
            # we need to return k+1 docs here as the top result is the input doc itself
            similarity_top_k=top_k + 1,
        )
        # remove the query doc itself from results
        scores, doc_ids = scores[1:], doc_ids[1:]
        return [self.documents_list[doc_id] for doc_id in doc_ids]

    def get_adjascent(
        self, doc: Document, direction: str = "next"
    ) -> t.Optional[Document]:
        # linear search for doc_id of doc in documents_list
        index = self.documents_list.index(doc)

        if direction == "next":
            if len(self.documents_list) > index + 1:
                next_doc = self.documents_list[index + 1]
                if next_doc.filename == doc.filename:
                    return next_doc
                else:
                    return None
            else:
                return None
        if direction == "prev":
            if index > 0:
                prev_doc = self.documents_list[index - 1]
                if prev_doc.filename == doc.filename:
                    return prev_doc
                else:
                    return None
            else:
                return None
