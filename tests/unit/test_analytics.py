from __future__ import annotations

from ragas._analytics import add_userid


def test_add_userid():
    payload = {"a": 1, "b": 2}
    payload = add_userid(payload)
    assert payload["userid"] is not None
