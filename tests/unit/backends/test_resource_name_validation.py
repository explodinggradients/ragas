"""Regression tests for ``BaseBackend._validate_resource_name``.

Verifies that every built-in backend rejects bad dataset / experiment
names at the API edge and accepts conventional ones.

Closes the gap tracked in #2846.
"""

from __future__ import annotations

import pytest

from ragas.backends.base import BaseBackend
from ragas.backends.inmemory import InMemoryBackend
from ragas.backends.local_csv import LocalCSVBackend
from ragas.backends.local_jsonl import LocalJSONLBackend

# --- The validator itself ---------------------------------------------


@pytest.mark.parametrize("good", ["qa_smoke", "exp-2026", "A_B_1", "v1.2"])
def test_validate_resource_name_accepts_conventional_names(good):
    BaseBackend._validate_resource_name(good)


@pytest.mark.parametrize(
    "bad",
    ["", "   ", "\t\n", ".", "..", ".hidden", "a/b", "a\\b", "ok\x00bad", "/abs"],
)
def test_validate_resource_name_rejects_bad_names(bad):
    with pytest.raises(ValueError):
        BaseBackend._validate_resource_name(bad)


@pytest.mark.parametrize("bad", [None, 1, 1.5, ["name"], {"name": "x"}, b"bytes"])
def test_validate_resource_name_rejects_non_strings(bad):
    with pytest.raises(TypeError):
        BaseBackend._validate_resource_name(bad)


# --- Per-backend wiring ------------------------------------------------


def _bad_name_backend(backend, name):
    """Try every public load / save method with a bad name; all should raise."""

    with pytest.raises((ValueError, TypeError)):
        backend.save_dataset(name, [{"k": "v"}])
    with pytest.raises((ValueError, TypeError)):
        backend.save_experiment(name, [{"k": "v"}])
    with pytest.raises((ValueError, TypeError)):
        backend.load_dataset(name)
    with pytest.raises((ValueError, TypeError)):
        backend.load_experiment(name)


def test_inmemory_backend_rejects_empty_name():
    be = InMemoryBackend()
    _bad_name_backend(be, "")


def test_inmemory_backend_rejects_path_traversal():
    be = InMemoryBackend()
    _bad_name_backend(be, "../../escape")


def test_inmemory_backend_rejects_hidden_name():
    be = InMemoryBackend()
    _bad_name_backend(be, ".hidden")


def test_inmemory_backend_rejects_none_name():
    be = InMemoryBackend()
    with pytest.raises(TypeError):
        be.save_dataset(None, [])  # type: ignore[arg-type]


def test_local_csv_backend_rejects_path_traversal(tmp_path):
    be = LocalCSVBackend(root_dir=str(tmp_path))
    _bad_name_backend(be, "../escape")


def test_local_csv_backend_rejects_leading_dot(tmp_path):
    be = LocalCSVBackend(root_dir=str(tmp_path))
    _bad_name_backend(be, ".hidden")


def test_local_jsonl_backend_rejects_path_traversal(tmp_path):
    be = LocalJSONLBackend(root_dir=str(tmp_path))
    _bad_name_backend(be, "../escape")


def test_local_jsonl_backend_rejects_leading_dot(tmp_path):
    be = LocalJSONLBackend(root_dir=str(tmp_path))
    _bad_name_backend(be, ".hidden")


# --- Happy paths stay green -------------------------------------------


def test_inmemory_backend_happy_path_roundtrip():
    be = InMemoryBackend()
    be.save_dataset("qa_smoke", [{"q": "Q", "a": "A"}])
    be.save_experiment("exp_smoke", [{"metric": "f", "value": 0.9}])
    assert be.load_dataset("qa_smoke") == [{"q": "Q", "a": "A"}]
    assert be.load_experiment("exp_smoke") == [{"metric": "f", "value": 0.9}]
    assert be.list_datasets() == ["qa_smoke"]
    assert be.list_experiments() == ["exp_smoke"]


def test_local_csv_backend_happy_path_roundtrip(tmp_path):
    be = LocalCSVBackend(root_dir=str(tmp_path))
    rows = [{"user": "alice", "score": "0.95"}]
    be.save_dataset("qa_smoke", rows)
    assert be.load_dataset("qa_smoke") == rows
    be.save_experiment("exp_smoke", [{"metric": "f", "value": "0.9"}])
    assert be.load_experiment("exp_smoke") == [{"metric": "f", "value": "0.9"}]
    assert "qa_smoke" in be.list_datasets()
    assert "exp_smoke" in be.list_experiments()


def test_local_jsonl_backend_happy_path_roundtrip(tmp_path):
    be = LocalJSONLBackend(root_dir=str(tmp_path))
    rows = [{"metric": "f", "value": 0.9}]
    be.save_dataset("qa_smoke", rows)
    assert be.load_dataset("qa_smoke") == rows
    be.save_experiment("exp_smoke", [{"metric": "f", "value": 0.9}])
    assert be.load_experiment("exp_smoke") == [{"metric": "f", "value": 0.9}]
    assert "qa_smoke" in be.list_datasets()
    assert "exp_smoke" in be.list_experiments()
