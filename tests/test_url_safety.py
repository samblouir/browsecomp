import pytest

from browsecomp250.browser.safety import UnsafeURLError, validate_url_syntax


def test_blocks_localhost() -> None:
    with pytest.raises(UnsafeURLError):
        validate_url_syntax("http://localhost/admin")


def test_blocks_private_ip() -> None:
    with pytest.raises(UnsafeURLError):
        validate_url_syntax("http://127.0.0.1/")


def test_blocks_nonstandard_port() -> None:
    with pytest.raises(UnsafeURLError):
        validate_url_syntax("https://example.com:8443/")


def test_allows_public_https() -> None:
    host, port = validate_url_syntax("https://example.com/path")
    assert host == "example.com"
    assert port is None
