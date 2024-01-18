import os
import pickle
import typing as t

import pytest
from langchain.text_splitter import TokenTextSplitter
from langchain_core.embeddings import Embeddings

from ragas.testset.docstore import Document, InMemoryDocumentStore


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
    a1 = Document(doc_id="a1", page_content="a1", filename="a")
    a2 = Document(doc_id="a2", page_content="a2", filename="a")
    b = Document(doc_id="b", page_content="b", filename="b")

    fake_embeddings = FakeEmbeddings()
    splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=0)

    store = InMemoryDocumentStore(splitter=splitter, embeddings=fake_embeddings)
    store.documents_list = [a1, a2, b]

    assert store.get_adjascent(a1) == a2
    assert store.get_adjascent(a2, "prev") == a1
    assert store.get_adjascent(a2, "next") is None
    assert store.get_adjascent(b, "prev") is None

    # raise ValueError if doc not in store
    c = Document(doc_id="c", page_content="c", filename="c")
    pytest.raises(ValueError, store.get_adjascent, c)


def create_test_documents(with_embeddings=True):
    if with_embeddings:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_embs.pkl")
        with open(path, "rb") as f:
            embeddings: dict[str, t.Any] = pickle.load(f)
    else:
        from collections import defaultdict

        embeddings = defaultdict(lambda: None)
    a1 = Document(
        doc_id="a1", page_content="cat", filename="a", embedding=embeddings["cat"]
    )
    a2 = Document(
        doc_id="a2", page_content="mouse", filename="a", embedding=embeddings["mouse"]
    )
    b = Document(
        doc_id="b",
        page_content="solar_system",
        filename="b",
        embedding=embeddings["solar_system"],
    )

    return a1, a2, b


def test_similar_nodes():
    a1, a2, b = create_test_documents()

    fake_embeddings = FakeEmbeddings()
    splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=0)
    store = InMemoryDocumentStore(splitter=splitter, embeddings=fake_embeddings)
    store.documents_list = [a1, a2, b]
    store.embeddings_list = [d.embedding for d in store.documents_list]

    assert store.get_similar(a1)[0] == a2
    assert store.get_similar(a2)[0] == a1
    assert store.get_similar(b, threshold=0.9) == []
    assert len(store.get_similar(b, top_k=2, threshold=0)) == 2
    assert len(store.get_similar(b, top_k=0, threshold=0)) == 0


def test_similar_nodes_scaled():
    a1, a2, b = create_test_documents()
    fake_embeddings = FakeEmbeddings()
    splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=0)
    store = InMemoryDocumentStore(splitter=splitter, embeddings=fake_embeddings)
    store.documents_list = [a1, a2, b] + [b] * 100
    store.embeddings_list = [d.embedding for d in store.documents_list]

    assert len(store.get_similar(a1, top_k=3)) == 3
    assert store.get_similar(a1)[0] == a2
    assert store.get_similar(a2)[0] == a1


def test_docstore_add():
    a1, a2, b = create_test_documents()

    fake_embeddings = FakeEmbeddings()
    splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=0)
    store = InMemoryDocumentStore(splitter=splitter, embeddings=fake_embeddings)
    docs_added = []
    for doc in [a1, a2, b]:
        store.add(doc)
        docs_added.append(doc)
        assert store.documents_list == docs_added
        assert store.embeddings_list == [d.embedding for d in docs_added]

    assert store.get(a1.doc_id) == a1


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


def test_docstore_add_batch():
    # create a dummy embeddings with support for async aembed_query()
    fake_embeddings = FakeEmbeddings()
    store = InMemoryDocumentStore(splitter=None, embeddings=fake_embeddings)  # type: ignore

    # add documents in batch
    docs = create_test_documents(with_embeddings=False)
    store.add(docs)
    assert (
        store.documents_map[docs[0].doc_id].embedding
        == fake_embeddings.embeddings[docs[0].page_content]
    )
    # add documents in batch that have some embeddings
    c = Document(doc_id="c", page_content="c", filename="c", embedding=[0.0] * 768)
    d = Document(doc_id="d", page_content="d", filename="d", embedding=[0.0] * 768)
    store.add([c, d])

    # test get() and that embeddings are correct
    assert store.get(c.doc_id).embedding == [0.0] * 768
    assert store.get(d.doc_id).embedding == [0.0] * 768
    assert len(store.documents_list) == 5
    assert len(store.embeddings_list) == 5
    assert len(store.documents_map) == 5
