from __future__ import annotations

import heapq
import logging
import typing as t
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import numpy.typing as npt
from langchain.text_splitter import TextSplitter
from langchain_core.documents import Document as LCDocument
from langchain_core.pydantic_v1 import Field

from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.exceptions import ExceptionInRunner
from ragas.executor import Executor
from ragas.run_config import RunConfig

if t.TYPE_CHECKING:
    from llama_index.core.schema import Document as LlamaindexDocument

    from ragas.testset.extractor import Extractor

Embedding = t.Union[t.List[float], npt.NDArray[np.float64]]
logger = logging.getLogger(__name__)


class Document(LCDocument):
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    embedding: t.Optional[t.List[float]] = Field(default=None, repr=False)

    @property
    def filename(self):
        filename = self.metadata.get("filename")
        if filename is not None:
            filename = self.metadata["filename"]
        else:
            logger.info(
                "Document [ID: %s] has no filename, using `doc_id` instead", self.doc_id
            )
            filename = self.doc_id

        return filename

    @classmethod
    def from_langchain_document(cls, doc: LCDocument):
        doc_id = str(uuid.uuid4())
        return cls(
            page_content=doc.page_content,
            metadata=doc.metadata,
            doc_id=doc_id,
        )

    @classmethod
    def from_llamaindex_document(cls, doc: LlamaindexDocument):
        doc_id = str(uuid.uuid4())
        return cls(
            page_content=doc.text,
            metadata=doc.metadata,
            doc_id=doc_id,
        )

    def __eq__(self, other) -> bool:
        # if the doc_id's are same then the Document objects are same
        return self.doc_id == other.doc_id


class Direction(str, Enum):
    """
    Direction for getting adjascent nodes.
    """

    NEXT = "next"
    PREV = "prev"
    UP = "up"
    DOWN = "down"


class Node(Document):
    keyphrases: t.List[str] = Field(default_factory=list, repr=False)
    relationships: t.Dict[Direction, t.Any] = Field(default_factory=dict, repr=False)
    doc_similarity: t.Optional[float] = Field(default=None, repr=False)
    wins: int = 0

    @property
    def next(self):
        return self.relationships.get(Direction.NEXT)

    @property
    def prev(self):
        return self.relationships.get(Direction.PREV)


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

    def set_run_config(self, run_config: RunConfig):
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
    extractor: t.Optional[Extractor] = field(default=None, repr=False)
    embeddings: t.Optional[BaseRagasEmbeddings] = field(default=None, repr=False)
    nodes: t.List[Node] = field(default_factory=list)
    node_embeddings_list: t.List[Embedding] = field(default_factory=list)
    node_map: t.Dict[str, Node] = field(default_factory=dict)
    run_config: RunConfig = field(default_factory=RunConfig)

    def _embed_items(self, items: t.Union[t.Sequence[Document], t.Sequence[Node]]):
        ...

    def add_documents(self, docs: t.Sequence[Document], show_progress=True):
        """
        Add documents in batch mode.
        """
        assert self.embeddings is not None, "Embeddings must be set"

        # split documents with self.splitter into smaller nodes
        nodes = [
            Node.from_langchain_document(d)
            for d in self.splitter.transform_documents(docs)
        ]
        self.add_nodes(nodes, show_progress=show_progress)

    def add_nodes(self, nodes: t.Sequence[Node], show_progress=True):
        assert self.embeddings is not None, "Embeddings must be set"
        assert self.extractor is not None, "Extractor must be set"

        # NOTE: Adds everything in async mode for now.
        nodes_to_embed = {}
        nodes_to_extract = {}

        # get embeddings for the docs
        executor = Executor(
            desc="embedding nodes",
            keep_progress_bar=False,
            raise_exceptions=True,
            run_config=self.run_config,
        )
        result_idx = 0
        for i, n in enumerate(nodes):
            if n.embedding is None:
                nodes_to_embed.update({i: result_idx})
                executor.submit(
                    self.embeddings.embed_text,
                    n.page_content,
                    name=f"embed_node_task[{i}]",
                )
                result_idx += 1

            if not n.keyphrases:
                nodes_to_extract.update({i: result_idx})
                executor.submit(
                    self.extractor.extract,
                    n,
                    name=f"keyphrase-extraction[{i}]",
                )
                result_idx += 1

        results = executor.results()
        if not results:
            raise ExceptionInRunner()

        for i, n in enumerate(nodes):
            if i in nodes_to_embed.keys():
                n.embedding = results[nodes_to_embed[i]]
            if i in nodes_to_extract.keys():
                keyphrases = results[nodes_to_extract[i]]
                n.keyphrases = keyphrases

            if n.embedding is not None and n.keyphrases != []:
                self.nodes.append(n)
                self.node_map[n.doc_id] = n
                assert isinstance(
                    n.embedding, (list, np.ndarray)
                ), "Embedding must be list or np.ndarray"
                self.node_embeddings_list.append(n.embedding)

        self.calculate_nodes_docs_similarity()
        self.set_node_relataionships()

    def set_node_relataionships(self):
        for i, node in enumerate(self.nodes):
            if i > 0:
                prev_node = self.nodes[i - 1]
                if prev_node.filename == node.filename:
                    node.relationships[Direction.PREV] = prev_node
                    prev_node.relationships[Direction.NEXT] = node
                else:
                    node.relationships[Direction.PREV] = None
                    prev_node.relationships[Direction.NEXT] = None
            if i == len(self.nodes) - 1:
                node.relationships[Direction.NEXT] = None

    def calculate_nodes_docs_similarity(self):
        doc_embeddings = {}
        filename_ids = set(
            [node.filename for node in self.nodes if node.filename is not None]
        )
        node_ids = set([node.doc_id for node in self.nodes])

        if len(filename_ids) == len(node_ids):
            logger.warning("Filename and doc_id are the same for all nodes.")
            for node in self.nodes:
                node.doc_similarity = 1.0

        else:
            for file_id in filename_ids:
                nodes_embedding = np.array(
                    [node.embedding for node in self.nodes if node.filename == file_id]
                )
                nodes_embedding = nodes_embedding.reshape(len(nodes_embedding), -1)
                doc_embeddings[file_id] = np.mean(nodes_embedding, axis=0)

            for node in self.nodes:
                assert node.embedding is not None, "Embedding cannot be None"
                node.doc_similarity = similarity(
                    node.embedding, doc_embeddings[node.filename]
                )

    def get_node(self, node_id: str) -> Node:
        return self.node_map[node_id]

    def get_document(self, doc_id: str) -> Node:
        raise NotImplementedError

    def get_random_nodes(self, k=1, alpha=0.1) -> t.List[Node]:
        def adjustment_factor(wins, alpha):
            return np.exp(-alpha * wins)

        scores = [adjustment_factor(node.wins, alpha) for node in self.nodes]
        similarity_scores = [node.doc_similarity for node in self.nodes]
        prob = np.array(scores) * np.array(similarity_scores)
        prob = prob / np.sum(prob)

        nodes = self.run_config.rng.choice(
            np.array(self.nodes), size=k, p=prob
        ).tolist()

        for node in nodes:
            idx = self.nodes.index(node)
            self.nodes[idx].wins += 1

        return nodes

    def get_similar(
        self, node: Node, threshold: float = 0.7, top_k: int = 3
    ) -> t.Union[t.List[Document], t.List[Node]]:
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

    def set_run_config(self, run_config: RunConfig):
        if self.embeddings:
            self.embeddings.set_run_config(run_config)
        self.run_config = run_config
