"""Tests for http_get against the in-process MockTransport."""

from __future__ import annotations

import httpx
import pytest

from agentic_runner.tools._base import ToolInvocationError
from agentic_runner.tools.http_get import HttpGetTool, set_transport


def test_http_get_to_example() -> None:
    out = HttpGetTool().invoke({"url": "http://example.com/"})
    assert out["status"] == 200
    assert "OK" in out["body"]


def test_http_get_blocks_disallowed_host() -> None:
    with pytest.raises(ToolInvocationError):
        HttpGetTool().invoke({"url": "http://evil.test/"})


def test_http_get_blocks_unknown_scheme() -> None:
    with pytest.raises(ToolInvocationError):
        HttpGetTool().invoke({"url": "file:///etc/passwd"})


def test_http_get_propagates_transport_error() -> None:
    def boom(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated network down")

    set_transport(httpx.MockTransport(boom))
    try:
        with pytest.raises(ToolInvocationError):
            HttpGetTool().invoke({"url": "http://example.com/"})
    finally:
        set_transport(httpx.MockTransport(lambda r: httpx.Response(200, text=f"OK: {r.url}")))
