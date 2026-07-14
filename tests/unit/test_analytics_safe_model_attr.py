from __future__ import annotations

import types
import typing as t

import pytest

from ragas._analytics import EmbeddingUsageEvent, _safe_model_attr


class _FakeModel:
    """Stand-in for a wrapped embedding object whose ``.model`` is not a string.

    Mirrors what a loaded ``SentenceTransformer`` / ``CrossEncoder`` or a
    ``FastEmbedEmbeddings`` instance looks like when the wrapper inspects
    ``getattr(self.embeddings, "model", None)``.
    """

    def __init__(self, label: str) -> None:
        self.label = label

    def __str__(self) -> str:
        return f"<FakeModel {self.label}>"


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ("text-embedding-3-small", "text-embedding-3-small"),
        ("", ""),
        (42, "42"),
        (3.14, "3.14"),
        (_FakeModel("hf-sentence-transformer"), "<FakeModel hf-sentence-transformer>"),
        ([], "[]"),
        ({"a": 1}, "{'a': 1}"),
    ],
)
def test_safe_model_attr_coercion(value: t.Any, expected: t.Optional[str]) -> None:
    assert _safe_model_attr(value) == expected


def test_safe_model_attr_passthrough_string_is_identity() -> None:
    # Strings must come back unchanged so telemetry payloads are byte-identical
    # for the common case (no extra quotes / repr wrapping).
    s = "text-embedding-3-small"
    assert _safe_model_attr(s) is s


def test_safe_model_attr_missing_attribute_returns_none() -> None:
    # The wrapper's `getattr(self.embeddings, "model", None)` pattern returns
    # `None` when the attribute is absent; the helper must propagate that.
    assert _safe_model_attr(getattr(object(), "model", None)) is None


def test_embedding_usage_event_accepts_stringified_non_string_model() -> None:
    # Without the helper this would raise ``pydantic.ValidationError``
    # because the field is typed ``Optional[str]`` and a SentenceTransformer
    # / CrossEncoder / FastEmbedEmbeddings instance is not a string.
    payload = EmbeddingUsageEvent(
        provider="langchain",
        model=_safe_model_attr(_FakeModel("sentence-transformers/all-MiniLM-L6-v2")),
        embedding_type="legacy",
        num_requests=1,
        is_async=False,
    )

    assert payload.model == "<FakeModel sentence-transformers/all-MiniLM-L6-v2>"
    assert payload.provider == "langchain"


def test_safe_model_attr_routes_through_track(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: a non-string model attribute must not crash ``track()``.

    Without the helper, ``track(EmbeddingUsageEvent(model=<SentenceTransformer>, ...))``
    raises ``pydantic.ValidationError`` *before* the ``@silent`` decorator on
    ``track`` runs, propagating out of every embedding call. With the helper,
    telemetry sees a coerced string and the user's metric keeps working.
    """
    captured: t.List[EmbeddingUsageEvent] = []

    def fake_track(event: EmbeddingUsageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr("ragas.embeddings.base.track", fake_track)

    from ragas.embeddings.base import LangchainEmbeddingsWrapper

    fake_langchain_embeddings = types.SimpleNamespace(
        model=_FakeModel("fastembed"),
        embed_query=lambda text: [0.0],
        embed_documents=lambda texts: [[0.0] for _ in texts],
        aembed_query=None,  # type: ignore[assignment]
        aembed_documents=None,  # type: ignore[assignment]
    )

    wrapper = LangchainEmbeddingsWrapper(fake_langchain_embeddings)  # type: ignore[arg-type]
    wrapper.embed_query("hello")
    wrapper.embed_documents(["a", "b"])

    assert len(captured) == 2
    assert captured[0].model == "<FakeModel fastembed>"
    assert captured[1].model == "<FakeModel fastembed>"


def test_safe_model_attr_preserves_string_passthrough_through_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing callers passing a string model see byte-identical telemetry."""
    captured: t.List[EmbeddingUsageEvent] = []

    def fake_track(event: EmbeddingUsageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr("ragas.embeddings.base.track", fake_track)

    from ragas.embeddings.base import LangchainEmbeddingsWrapper

    fake_langchain_embeddings = types.SimpleNamespace(
        model="text-embedding-3-small",
        embed_query=lambda text: [0.0],
        embed_documents=lambda texts: [[0.0] for _ in texts],
        aembed_query=None,  # type: ignore[assignment]
        aembed_documents=None,  # type: ignore[assignment]
    )

    wrapper = LangchainEmbeddingsWrapper(fake_langchain_embeddings)  # type: ignore[arg-type]
    wrapper.embed_query("hello")

    assert captured[0].model == "text-embedding-3-small"


def test_safe_model_attr_callable_helper_returns_str() -> None:
    """A callable that defines ``__str__`` is coerced to its string form."""

    class _Callable:
        def __call__(self) -> str:
            return "called"

        def __str__(self) -> str:
            return "<Callable>"

    assert _safe_model_attr(_Callable()) == "<Callable>"
