import socket

from ragas.prompt.multi_modal_prompt import ImageTextPromptValue


def _fake_loopback_getaddrinfo(host, port, family=socket.AF_UNSPEC):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))]


def test_download_validate_and_encode_blocks_backslash_ssrf(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _fake_loopback_getaddrinfo)

    def fail_requests_get(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("requests.get should not be called for unsafe URLs")

    monkeypatch.setattr(
        "ragas.prompt.multi_modal_prompt.requests.get", fail_requests_get
    )

    prompt_value = ImageTextPromptValue(items=[])
    assert (
        prompt_value._download_validate_and_encode("http://127.0.0.1:6666\\@1.1.1.1")
        is None
    )
