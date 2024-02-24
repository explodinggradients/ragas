import uuid

from ragas.testset.docstore import Document


def test_document_filename(monkeypatch):
    monkeypatch.setattr(uuid, "uuid4", lambda: "test-uuid")
    d1 = Document(page_content="a1")
    assert d1.filename == "test-uuid"

    # now suppose I add a filename to metadata
    d2 = Document(page_content="a2", metadata={"filename": "test-filename"})
    assert d2.filename == "test-filename"


def test_document_chunking():
    """
    Tests to make sure that there is no problem when you chunk a document into Nodes
    especially because of the fact that Node objects are created again.
    """
    from langchain.text_splitter import TokenTextSplitter
    from langchain_core.documents import Document

    from ragas.testset.docstore import Node

    splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)
    doc = Document(page_content="Hello, world!", metadata={"filename": "test-filename"})
    nodes = [
        Node.from_langchain_document(d) for d in splitter.transform_documents([doc])
    ]
    for node in nodes:
        assert node.filename == "test-filename"
