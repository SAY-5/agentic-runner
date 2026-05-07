"""Tests for the tool registry — invocation, validation, timeout."""

from __future__ import annotations

import time
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel

from agentic_runner.tools._base import (
    REGISTRY,
    ToolInvocationError,
    ToolTimeoutError,
    invoke_with_guard,
    list_tool_specs,
    register_tool,
)


class _SlowInput(BaseModel):
    pass


class _SlowOutput(BaseModel):
    ok: bool


def test_registry_contains_expected_tools() -> None:
    for name in [
        "calculate",
        "query_db",
        "read_file",
        "write_file",
        "http_get",
        "summarize",
        "extract_json",
        "finish",
    ]:
        assert name in REGISTRY


def test_list_tool_specs_function_calling_shape() -> None:
    specs = list_tool_specs()
    by_name = {s["name"]: s for s in specs}
    spec = by_name["calculate"]
    assert "input_schema" in spec
    assert spec["input_schema"]["type"] == "object"


def test_invoke_with_guard_propagates_invocation_error() -> None:
    tool = REGISTRY["calculate"]
    with pytest.raises(ToolInvocationError):
        invoke_with_guard(tool, {"expression": "x"})


def test_invoke_with_guard_enforces_timeout() -> None:
    @register_tool
    class _SlowTool:
        name: ClassVar[str] = "slow_test_tool_unique_name"
        description: ClassVar[str] = "deliberately slow"
        input_model: ClassVar[type[BaseModel]] = _SlowInput
        output_model: ClassVar[type[BaseModel]] = _SlowOutput
        max_runtime_ms: ClassVar[int] = 5
        idempotent: ClassVar[bool] = True

        def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
            time.sleep(0.05)
            return {"ok": True}

    tool = REGISTRY["slow_test_tool_unique_name"]
    try:
        with pytest.raises(ToolTimeoutError):
            invoke_with_guard(tool, {})
    finally:
        REGISTRY.pop("slow_test_tool_unique_name", None)


def test_register_tool_rejects_duplicate() -> None:
    with pytest.raises(ValueError):
        from agentic_runner.tools.calculate import CalculateTool

        register_tool(CalculateTool)
