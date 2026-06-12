import socket

from ragas.url_safety import prepare_safe_http_url


def _fake_public_getaddrinfo(host, port, family=socket.AF_UNSPEC):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]


def _fake_loopback_getaddrinfo(host, port, family=socket.AF_UNSPEC):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))]


def test_prepare_safe_http_url_allows_public_http_url(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _fake_public_getaddrinfo)

    assert (
        prepare_safe_http_url("http://example.com/image.png")
        == "http://example.com/image.png"
    )


def test_prepare_safe_http_url_blocks_loopback_url(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _fake_loopback_getaddrinfo)

    assert prepare_safe_http_url("http://127.0.0.1:6666/image.png") is None


def test_prepare_safe_http_url_blocks_backslash_bypass(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _fake_loopback_getaddrinfo)

    assert prepare_safe_http_url("http://127.0.0.1:6666\\@1.1.1.1") is None
