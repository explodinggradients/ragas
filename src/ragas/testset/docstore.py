import heapq
import typing as t
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from random import choices

import numpy as np
import numpy.typing as npt
from langchain.text_splitter import TextSplitter
from langchain_core.documents import Document as LCDocument
from llama_index.readers.schema import Document as LlamaindexDocument
from langchain_core.pydantic_v1 import Field

from ragas.async_utils import run_async_tasks
from ragas.embeddings.base import BaseRagasEmbeddings, embedding_factory

Embedding = t.Union[t.List[float], npt.NDArray[np.float64]]
logger = logging.getLogger(__name__)
rng = np.random.default_rng()


class Document(LCDocument):
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: t.Optional[str] = None
    embedding: t.Optional[t.List[float]] = Field(default=None, repr=False)

    @classmethod
    def from_langchain_document(cls, doc: LCDocument):
        doc_id = str(uuid.uuid4())
        if doc.metadata.get("filename"):
            filename = doc.metadata["filename"]
        else:
            logger.info(
                "Document [ID: %s] has no filename. Using doc_id as filename.", doc_id
            )
            filename = doc_id
        return cls(
            page_content=doc.page_content,
            metadata=doc.metadata,
            doc_id=doc_id,
            filename=filename,
        )

    @classmethod
    def from_llamaindex_document(cls, doc: LlamaindexDocument):
        doc_id = str(uuid.uuid4())
        if doc.metadata.get("filename"):
            filename = doc.metadata["filename"]
        else:
            logger.info(
                "Document [ID: %s] has no filename. Using doc_id as filename.", doc_id
            )
            filename = doc_id
        return cls(
            page_content=doc.text,
            metadata=doc.metadata,
            doc_id=doc_id,
            filename=filename,
        )


class Node(Document):
    ...


class Direction(str, Enum):
    """
    Direction for getting adjascent nodes.
    """

    NEXT = "next"
    PREV = "prev"
    UP = "up"
    DOWN = "down"


class DocumentStore(ABC):
    def __init__(self):
        self.documents = {}

    @abstractmethod
    def add_documents(self, docs: t.Sequence[Document], show_progress=True):
        ...

    @abstractmethod
    def add_nodes(self, nodes: t.Sequence[Node], show_progress=True):
        ...

    @abstractmethod
    def get_node(self, node_id: str) -> Node:
        ...

    @abstractmethod
    def get_random_nodes(self, k=1) -> t.List[Node]:
        ...

    @abstractmethod
    def get_similar(
        self, node: Node, threshold: float = 0.7, top_k: int = 3
    ) -> t.Union[t.List[Document], t.List[Node]]:
        ...

    @abstractmethod
    def get_adjascent(
        self, node: Node, direction: Direction = Direction.NEXT
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
    nodes: t.List[Node] = field(default_factory=list)
    node_embeddings_list: t.List[Embedding] = field(default_factory=list)
    node_map: t.Dict[str, Node] = field(default_factory=dict)

    def _embed_items(self, items: t.Union[t.Sequence[Document], t.Sequence[Node]]):
        ...

    def add_documents(self, docs: t.Sequence[Document], show_progress=True):
        """
        Add documents in batch mode.
        """
        # split documents with self.splitter into smaller nodes
        nodes = [
            Node.from_langchain_document(d)
            for d in self.splitter.transform_documents(docs)
        ]

        self.add_nodes(nodes, show_progress=show_progress)

    def add_nodes(self, nodes: t.Sequence[Node], show_progress=True):
        # NOTE: Adds everything in async mode for now.
        embed_tasks = []
        docs_to_embed = []
        # get embeddings for the docs
        for n in nodes:
            if n.embedding is None:
                embed_tasks.append(self.embeddings.aembed_query(n.page_content))
                docs_to_embed.append(n)
            else:
                self.nodes.append(n)
                self.node_map[n.doc_id] = n
                self.node_embeddings_list.append(n.embedding)

        embeddings = run_async_tasks(embed_tasks, show_progress=show_progress)
        for n, embedding in zip(docs_to_embed, embeddings):
            n.embedding = embedding
            self.nodes.append(n)
            self.node_map[n.doc_id] = n
            self.node_embeddings_list.append(n.embedding)

    def get_node(self, node_id: str) -> Node:
        return self.node_map[node_id]

    def get_document(self, doc_id: str) -> Node:
        raise NotImplementedError

    def get_random_nodes(self, k=1) -> t.List[Node]:
        return choices(self.nodes, k=k)

    def get_similar(
        self, node: Node, threshold: float = 0.7, top_k: int = 3
    ) -> t.Union[t.List[Document], t.List[Node]]:
        items = []
        doc = node
        if doc.embedding is None:
            raise ValueError("Document has no embedding.")
        scores, doc_ids = get_top_k_embeddings(
            query_embedding=doc.embedding,
            embeddings=self.node_embeddings_list,
            similarity_fn=similarity,
            similarity_cutoff=threshold,
            # we need to return k+1 docs here as the top result is the input doc itself
            similarity_top_k=top_k + 1,
        )
        # remove the query doc itself from results
        scores, doc_ids = scores[1:], doc_ids[1:]
        items = [self.nodes[doc_id] for doc_id in doc_ids]
        return items

    def get_adjascent(
        self, node: Node, direction: Direction = Direction.NEXT
    ) -> t.Optional[Document]:
        # linear search for doc_id of doc in documents_list
        index = self.nodes.index(node)

        if direction == Direction.NEXT:
            if len(self.nodes) > index + 1:
                next_doc = self.nodes[index + 1]
                if next_doc.filename == node.filename:
                    return next_doc
                else:
                    return None
            else:
                return None
        if direction == Direction.PREV:
            if index > 0:
                prev_doc = self.nodes[index - 1]
                if prev_doc.filename == node.filename:
                    return prev_doc
                else:
                    return None
            else:
                return None
