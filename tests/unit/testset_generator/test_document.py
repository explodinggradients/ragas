import uuid
from ragas.testset.docstore import Document


def test_document_filename(monkeypatch):
    monkeypatch.setattr(uuid, "uuid4", lambda: "test-uuid")
    d1 = Document(page_content="a1")
    assert d1.filename == "test-uuid"

    # now suppose I add a filename to metadata
    d2 = Document(page_content="a2", metadata={"filename": "test-filename"})
    assert d2.filename == "test-filename"

    # can I change the filename?
    d2.filename = "new-filename"
    assert d2.filename == "new-filename"
