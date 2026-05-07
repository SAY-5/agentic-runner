"""Tests for sandboxed read_file/write_file."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_runner.tools._base import ToolInvocationError
from agentic_runner.tools.read_file import ReadFileTool
from agentic_runner.tools.write_file import WriteFileTool


def test_write_then_read_roundtrip(tmp_workspace: Path) -> None:
    out = WriteFileTool().invoke({"path": "hello.txt", "content": "world"})
    assert out["bytes_written"] == 5
    out2 = ReadFileTool().invoke({"path": "hello.txt"})
    assert out2["content"] == "world"
    assert out2["bytes"] == 5


def test_read_blocks_path_traversal(tmp_workspace: Path) -> None:
    with pytest.raises(ToolInvocationError):
        ReadFileTool().invoke({"path": "../etc/passwd"})


def test_write_blocks_path_traversal(tmp_workspace: Path) -> None:
    with pytest.raises(ToolInvocationError):
        WriteFileTool().invoke({"path": "../escape.txt", "content": "x"})


def test_read_missing_file(tmp_workspace: Path) -> None:
    with pytest.raises(ToolInvocationError):
        ReadFileTool().invoke({"path": "nope.txt"})


def test_read_size_cap(tmp_workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTIC_RUNNER_MAX_FILE_BYTES", "10")
    from agentic_runner.settings import reset_settings_cache

    reset_settings_cache()

    WriteFileTool().invoke({"path": "ok.txt", "content": "1234567"})
    big = "0" * 64
    with pytest.raises(ToolInvocationError):
        WriteFileTool().invoke({"path": "big.txt", "content": big})
