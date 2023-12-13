import os
import pickle
import typing as t

import pytest

from ragas.testset.docstore import Document, InMemoryDocumentStore


def test_adjacent_nodes():
    a1 = Document(doc_id="a1", page_content="a1", filename="a")
    a2 = Document(doc_id="a2", page_content="a2", filename="a")
    b = Document(doc_id="b", page_content="b", filename="b")

    store = InMemoryDocumentStore(splitter=None)
    store.documents_list = [a1, a2, b]

    assert store.get_adjascent(a1) == a2
    assert store.get_adjascent(a2, "prev") == a1
    assert store.get_adjascent(a2, "next") == None
    assert store.get_adjascent(b, "prev") == None

    # raise ValueError if doc not in store
    c = Document(doc_id="c", page_content="c", filename="c")
    pytest.raises(ValueError, store.get_adjascent, c)


def create_test_documents():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_embs.pkl")
    with open(path, "rb") as f:
        embeddings: dict[str, t.Any] = pickle.load(f)
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
    store = InMemoryDocumentStore(splitter=None)
    store.documents_list = [a1, a2, b]
    store.embeddings_list = [d.embedding for d in store.documents_list]

    assert store.get_similar(a1)[0] == a2
    assert store.get_similar(a2)[0] == a1
    assert store.get_similar(b, threshold=0.9) == []
    assert len(store.get_similar(b, top_k=2, threshold=0)) == 2
    assert len(store.get_similar(b, top_k=0, threshold=0)) == 0


def test_similar_nodes_scaled():
    a1, a2, b = create_test_documents()
    store = InMemoryDocumentStore(splitter=None)
    store.documents_list = [a1, a2, b] + [b] * 100
    store.embeddings_list = [d.embedding for d in store.documents_list]

    assert len(store.get_similar(a1, top_k=3)) == 3
    assert store.get_similar(a1)[0] == a2
    assert store.get_similar(a2)[0] == a1
