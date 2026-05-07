"""HTTP GET against an allowlist of hosts."""

from __future__ import annotations

from typing import Any, ClassVar
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from agentic_runner.settings import get_settings
from agentic_runner.tools._base import ToolInvocationError, register_tool


class HttpGetInput(BaseModel):
    url: str = Field(min_length=4, max_length=1024)


class HttpGetOutput(BaseModel):
    status: int
    body: str
    headers: dict[str, str]


def _mock_handler(request: httpx.Request) -> httpx.Response:
    """Default mock handler returning a deterministic 200 OK page."""
    return httpx.Response(
        200,
        headers={"content-type": "text/plain"},
        text=f"OK: {request.url}",
    )


_TRANSPORT: httpx.BaseTransport = httpx.MockTransport(_mock_handler)


def set_transport(transport: httpx.BaseTransport) -> None:
    """Swap the HTTP transport (used for tests / live mode)."""
    global _TRANSPORT
    _TRANSPORT = transport


@register_tool
class HttpGetTool:
    name: ClassVar[str] = "http_get"
    description: ClassVar[str] = "Fetch a URL via HTTP GET, restricted to an allowlist of hosts."
    input_model: ClassVar[type[BaseModel]] = HttpGetInput
    output_model: ClassVar[type[BaseModel]] = HttpGetOutput
    max_runtime_ms: ClassVar[int] = 2000
    idempotent: ClassVar[bool] = True

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        parsed = HttpGetInput.model_validate(args)
        u = urlparse(parsed.url)
        if u.scheme not in {"http", "https"}:
            raise ToolInvocationError(f"http_get: scheme not allowed: {u.scheme}")
        host = (u.hostname or "").lower()
        allowlist = {h.lower() for h in get_settings().http_allowlist}
        if host not in allowlist:
            raise ToolInvocationError(f"http_get: host not on allowlist: {host}")

        try:
            with httpx.Client(transport=_TRANSPORT, timeout=2.0) as client:
                resp = client.get(parsed.url)
        except httpx.HTTPError as exc:
            raise ToolInvocationError(f"http_get: {exc}") from exc

        return {
            "status": resp.status_code,
            "body": resp.text[: get_settings().max_file_bytes],
            "headers": {k.lower(): v for k, v in resp.headers.items()},
        }
