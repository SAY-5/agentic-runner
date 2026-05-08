"""Tests for the ComposedTool primitive and the sample summarize_document."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel, Field, ValidationError

from agentic_runner.tools import REGISTRY, ComposedStep, ComposedTool, ToolInvocationError
from agentic_runner.tools._base import Tool, register_tool

# ---------------------------------------------------------------------------
# Sample composed tool surface
# ---------------------------------------------------------------------------


def test_composed_tool_satisfies_tool_protocol() -> None:
    inst = REGISTRY["summarize_document"]
    assert isinstance(inst, Tool)
    assert inst.name == "summarize_document"
    assert inst.input_model is not None
    assert inst.output_model is not None
    # The composed tool's own schema is independent of any inner step.
    schema = inst.input_model.model_json_schema()
    assert "path" in schema["properties"]
    assert "max_words" in schema["properties"]


def test_composed_tool_input_schema_validates() -> None:
    inst = REGISTRY["summarize_document"]
    with pytest.raises(ValidationError):
        inst.input_model.model_validate({})  # missing path


def test_composed_tool_output_schema_validates() -> None:
    inst = REGISTRY["summarize_document"]
    payload = {
        "summary": "ok",
        "source_path": "x.txt",
        "source_bytes": 10,
        "word_count": 1,
    }
    inst.output_model.model_validate(payload)
    with pytest.raises(ValidationError):
        inst.output_model.model_validate({"summary": "ok"})


# ---------------------------------------------------------------------------
# Successful execution of the inner sequence
# ---------------------------------------------------------------------------


def test_summarize_document_executes_inner_sequence(tmp_workspace: Path) -> None:
    (tmp_workspace / "note.txt").write_text(
        "The roadmap covers infrastructure and observability investments.",
        encoding="utf-8",
    )
    inst = REGISTRY["summarize_document"]
    out = inst.invoke({"path": "note.txt", "max_words": 20})
    assert out["source_path"] == "note.txt"
    assert out["source_bytes"] > 0
    assert out["summary"]
    assert out["word_count"] >= 1


def test_summarize_document_propagates_read_failure(tmp_workspace: Path) -> None:
    inst = REGISTRY["summarize_document"]
    with pytest.raises(ToolInvocationError, match="read_file"):
        inst.invoke({"path": "does_not_exist.txt", "max_words": 10})


# ---------------------------------------------------------------------------
# Custom ComposedTool to exercise general failure propagation paths
# ---------------------------------------------------------------------------


class _CalcThenFinishInput(BaseModel):
    expr_a: str = Field(min_length=1)
    expr_b: str = Field(min_length=1)


class _CalcThenFinishOutput(BaseModel):
    sum_value: float
    final: str


def _ca(parsed: dict[str, Any], _prior: list[dict[str, Any]]) -> dict[str, Any]:
    return {"expression": parsed["expr_a"]}


def _cb(_parsed: dict[str, Any], prior: list[dict[str, Any]]) -> dict[str, Any]:
    return {"expression": f"{prior[0]['value']}+1"}


def _cfin(_parsed: dict[str, Any], prior: list[dict[str, Any]]) -> dict[str, Any]:
    return {"result": f"sum={prior[0]['value']}+{prior[1]['value']}"}


def _combine(parsed: dict[str, Any], prior: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sum_value": prior[0]["value"] + prior[1]["value"],
        "final": prior[2]["final"],
    }


@register_tool
class _CalcThenFinishTool(ComposedTool):
    """Composed: calculate -> calculate -> finish.

    Used only by tests — it isn't exposed to the planner because the
    selector pulls names off ``REGISTRY`` and the eval suite never names
    this tool.
    """

    name: ClassVar[str] = "test_calc_chain"
    description: ClassVar[str] = "Test composed tool chaining two calculate steps."
    input_model: ClassVar[type[BaseModel]] = _CalcThenFinishInput
    output_model: ClassVar[type[BaseModel]] = _CalcThenFinishOutput
    max_runtime_ms: ClassVar[int] = 1_000
    idempotent: ClassVar[bool] = True

    steps = [
        ComposedStep(tool_name="calculate", args_builder=_ca),
        ComposedStep(tool_name="calculate", args_builder=_cb),
        ComposedStep(tool_name="finish", args_builder=_cfin),
    ]
    output_combiner = staticmethod(_combine)


def test_custom_composed_tool_chains_two_calculates() -> None:
    inst = REGISTRY["test_calc_chain"]
    out = inst.invoke({"expr_a": "2+2", "expr_b": "ignored"})
    assert out["sum_value"] == pytest.approx(4 + 5)  # 4 from first, 4+1=5 from second
    assert out["final"] == "sum=4.0+5.0"


def test_custom_composed_tool_propagates_inner_failure() -> None:
    inst = REGISTRY["test_calc_chain"]
    with pytest.raises(ToolInvocationError, match="calculate"):
        inst.invoke({"expr_a": "syntax !!!error", "expr_b": "ignored"})


def test_composed_tool_fails_on_unregistered_inner_step() -> None:
    """A composed tool that references an unknown inner tool fails with a
    typed ToolInvocationError that names the missing tool."""

    class _BogusInput(BaseModel):
        x: str

    class _BogusOutput(BaseModel):
        ok: bool

    bogus = ComposedTool()
    bogus.name = "bogus_inline"
    bogus.description = "test only"
    bogus.input_model = _BogusInput
    bogus.output_model = _BogusOutput
    bogus.steps = [
        ComposedStep(
            tool_name="this_tool_does_not_exist",
            args_builder=lambda p, q: {},
        )
    ]
    bogus.output_combiner = staticmethod(lambda p, q: {"ok": True})

    with pytest.raises(ToolInvocationError, match="this_tool_does_not_exist"):
        bogus.invoke({"x": "y"})


def test_composed_tool_validates_combined_output() -> None:
    """If the combiner yields a payload that does not match the declared
    output_model, that failure surfaces as a ValidationError (not a silent
    pass)."""

    class _StrictIn(BaseModel):
        path: str

    class _StrictOut(BaseModel):
        required_field: int

    bad = ComposedTool()
    bad.name = "bad_combiner_inline"
    bad.description = "test only"
    bad.input_model = _StrictIn
    bad.output_model = _StrictOut
    bad.steps = []
    bad.output_combiner = staticmethod(lambda p, q: {"wrong_field": "x"})

    with pytest.raises(ValidationError):
        bad.invoke({"path": "p"})
