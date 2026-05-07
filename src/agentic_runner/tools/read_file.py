"""Sandboxed file read confined to the workspace directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from agentic_runner.settings import get_settings
from agentic_runner.tools._base import ToolInvocationError, register_tool


class ReadFileInput(BaseModel):
    path: str = Field(min_length=1, max_length=512)


class ReadFileOutput(BaseModel):
    content: str
    bytes: int


def _resolve_safe(rel: str) -> Path:
    settings = get_settings()
    root = settings.workspace_dir.resolve()
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ToolInvocationError(f"read_file: path escapes workspace sandbox: {rel!r}") from exc
    return candidate


@register_tool
class ReadFileTool:
    name: ClassVar[str] = "read_file"
    description: ClassVar[str] = "Read a UTF-8 text file from the sandboxed workspace directory."
    input_model: ClassVar[type[BaseModel]] = ReadFileInput
    output_model: ClassVar[type[BaseModel]] = ReadFileOutput
    max_runtime_ms: ClassVar[int] = 200
    idempotent: ClassVar[bool] = True

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        parsed = ReadFileInput.model_validate(args)
        target = _resolve_safe(parsed.path)
        if not target.is_file():
            raise ToolInvocationError(f"read_file: not a file: {parsed.path}")
        max_bytes = get_settings().max_file_bytes
        size = target.stat().st_size
        if size > max_bytes:
            raise ToolInvocationError(f"read_file: file exceeds max bytes ({size} > {max_bytes})")
        content = target.read_text(encoding="utf-8")
        return {"content": content, "bytes": size}
