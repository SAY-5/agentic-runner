"""Sandboxed file write confined to the workspace directory."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, Field

from agentic_runner.settings import get_settings
from agentic_runner.tools._base import ToolInvocationError, register_tool
from agentic_runner.tools.read_file import _resolve_safe


class WriteFileInput(BaseModel):
    path: str = Field(min_length=1, max_length=512)
    content: str = Field(default="", max_length=200_000)


class WriteFileOutput(BaseModel):
    bytes_written: int


@register_tool
class WriteFileTool:
    name: ClassVar[str] = "write_file"
    description: ClassVar[str] = "Write a UTF-8 text file under the sandboxed workspace directory."
    input_model: ClassVar[type[BaseModel]] = WriteFileInput
    output_model: ClassVar[type[BaseModel]] = WriteFileOutput
    max_runtime_ms: ClassVar[int] = 300
    idempotent: ClassVar[bool] = False

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        parsed = WriteFileInput.model_validate(args)
        target = _resolve_safe(parsed.path)
        max_bytes = get_settings().max_file_bytes
        encoded = parsed.content.encode("utf-8")
        if len(encoded) > max_bytes:
            raise ToolInvocationError(
                f"write_file: content exceeds max bytes ({len(encoded)} > {max_bytes})"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(encoded)
        return {"bytes_written": len(encoded)}
