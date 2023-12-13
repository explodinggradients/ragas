from ragas.testset.docstore import InMemoryDocumentStore
from ragas.testset.nodes import Document


def test_adjacent_nodes():
    a1 = Document(doc_id="a1", page_content="a1", filename="a")
    a2 = Document(doc_id="a2", page_content="a2", filename="a")
    b = Document(doc_id="b", page_content="b", filename="b")

    store = InMemoryDocumentStore(splitter=None)
    store.documents = [a1, a2, b]

    assert store.get_adjascent(a1) == a2
    assert store.get_adjascent(a2, "prev") == a1
    assert store.get_adjascent(a2, "next") == None
    assert store.get_adjascent(b, "prev") == None


def test_similar_nodes():
    a1 = Document(doc_id="a1", page_content="cat", filename="a")
    a2 = Document(doc_id="a2", page_content="mouse", filename="a")
    b = Document(doc_id="b", page_content="table", filename="b")

    store = InMemoryDocumentStore(splitter=None)
    store.documents = [a1, a2, b]

    assert store.get_similar(a1) == []
    assert store.get_similar(a2) == []
    assert store.get_similar(b) == []
