import os
import pickle
import typing as t

import numpy as np
import pytest
from langchain.text_splitter import TokenTextSplitter
from langchain_core.embeddings import Embeddings

from ragas.testset.docstore import InMemoryDocumentStore, Node


class FakeEmbeddings(Embeddings):
    def __init__(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_embs.pkl")
        with open(path, "rb") as f:
            self.embeddings: dict[str, t.Any] = pickle.load(f)

    def _get_embedding(self, text: str) -> t.List[float]:
        if text in self.embeddings:
            return self.embeddings[text]
        else:
            return [0] * 768

    def embed_documents(self, texts: t.List[str]) -> t.List[t.List[float]]:
        return [self._get_embedding(text) for text in texts]

    def embed_query(self, text: str) -> t.List[float]:
        return self._get_embedding(text)

    async def aembed_query(self, text: str) -> t.List[float]:
        return self._get_embedding(text)


def test_adjacent_nodes():
    a1 = Node(doc_id="a1", page_content="a1", metadata={"filename": "a"})
    a2 = Node(doc_id="a2", page_content="a2", metadata={"filename": "a"})
    b = Node(doc_id="b", page_content="b", metadata={"filename": "a"})

    fake_embeddings = FakeEmbeddings()
    splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=0)
    store = InMemoryDocumentStore(splitter=splitter, embeddings=fake_embeddings)
    store.nodes = [a1, a2, b]
    store.set_node_relataionships()

    assert store.nodes[0].next == a2
    assert store.nodes[1].prev == a1
    assert store.nodes[2].next is None


def create_test_nodes(with_embeddings=True):
    if with_embeddings:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_embs.pkl")
        with open(path, "rb") as f:
            embeddings: dict[str, t.Any] = pickle.load(f)
    else:
        from collections import defaultdict

        embeddings = defaultdict(lambda: None)
    a1 = Node(
        doc_id="a1",
        page_content="cat",
        metadata={"filename": "a"},
        embedding=embeddings["cat"],
    )
    a2 = Node(
        doc_id="a2",
        page_content="mouse",
        metadata={"filename": "a"},
        embedding=embeddings["mouse"],
    )
    b = Node(
        doc_id="b",
        page_content="solar_system",
        filename="b",
        embedding=embeddings["solar_system"],
    )

    return a1, a2, b


def test_similar_nodes():
    a1, a2, b = create_test_nodes()
    fake_embeddings = FakeEmbeddings()
    splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=0)
    store = InMemoryDocumentStore(splitter=splitter, embeddings=fake_embeddings)
    store.nodes = [a1, a2, b]
    store.node_embeddings_list = [d.embedding for d in store.nodes]

    assert store.get_similar(a1)[0] == a2
    assert store.get_similar(a2)[0] == a1
    assert store.get_similar(b, threshold=0.9) == []
    assert len(store.get_similar(b, top_k=2, threshold=0)) == 2
    assert len(store.get_similar(b, top_k=0, threshold=0)) == 0


def test_similar_nodes_scaled():
    a1, a2, b = create_test_nodes()
    fake_embeddings = FakeEmbeddings()
    splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=0)
    store = InMemoryDocumentStore(splitter=splitter, embeddings=fake_embeddings)
    store.nodes = [a1, a2, b] + [b] * 100
    store.node_embeddings_list = [d.embedding for d in store.nodes]

    assert len(store.get_similar(a1, top_k=3)) == 3
    assert store.get_similar(a1)[0] == a2
    assert store.get_similar(a2)[0] == a1


@pytest.fixture
def test_docstore_add(fake_llm):
    a1, a2, b = create_test_nodes()

    fake_embeddings = FakeEmbeddings()
    splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=0)
    store = InMemoryDocumentStore(
        splitter=splitter, embeddings=fake_embeddings, llm=fake_llm
    )
    docs_added = []
    for doc in [a1, a2, b]:
        store.add_nodes([doc])
        docs_added.append(doc)
        assert store.nodes == docs_added
        assert store.node_embeddings_list == [d.embedding for d in docs_added]
        assert np.all([node.keyphrases != [] for node in store.nodes])

    assert store.get_node(a1.doc_id) == a1


@pytest.mark.asyncio
async def test_fake_embeddings():
    fake_embeddings = FakeEmbeddings()
    assert fake_embeddings.embed_query("cat") == fake_embeddings.embeddings["cat"]
    assert fake_embeddings.embed_query("cat") != fake_embeddings.embeddings["mouse"]
    assert fake_embeddings.embed_documents(["cat", "mouse"]) == [
        fake_embeddings.embeddings["cat"],
        fake_embeddings.embeddings["mouse"],
    ]
    assert (
        await fake_embeddings.aembed_query("cat") == fake_embeddings.embeddings["cat"]
    )


@pytest.fixture
def test_docstore_add_batch(fake_llm):
    # create a dummy embeddings with support for async aembed_query()
    fake_embeddings = FakeEmbeddings()
    store = InMemoryDocumentStore(
        splitter=None, embeddings=fake_embeddings, llm=fake_llm
    )  # type: ignore

    # add documents in batch
    nodes = create_test_nodes(with_embeddings=False)
    store.add_nodes(nodes)
    assert (
        store.node_map[nodes[0].doc_id].embedding
        == fake_embeddings.embeddings[nodes[0].page_content]
    )
    # add documents in batch that have some embeddings
    c = Node(
        doc_id="c", page_content="c", metadata={"filename": "c"}, embedding=[0.0] * 768
    )
    d = Node(
        doc_id="d", page_content="d", metadata={"filename": "d"}, embedding=[0.0] * 768
    )
    store.add_nodes([c, d])

    # test get() and that embeddings and keyphrases are correct
    assert store.get_node(c.doc_id).embedding == [0.0] * 768
    assert len(store.get_node(c.doc_id).keyphrases) == 1
    assert len(store.get_node(d.doc_id).keyphrases) == 1
    assert store.get_node(d.doc_id).embedding == [0.0] * 768
    assert len(store.nodes) == 5
    assert len(store.node_embeddings_list) == 5
    assert len(store.node_map) == 5
