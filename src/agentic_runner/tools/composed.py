"""Composed tools — wrap a fixed multi-tool sequence as a single primitive.

A ``ComposedTool`` lets a domain-specific user package a deterministic
sequence of inner tool invocations as one logical action that satisfies
the :class:`agentic_runner.tools._base.Tool` protocol. The runner sees a
single tool call; internally the composed tool executes its inner steps
in order, threads each step's output forward, and produces a combined
output payload validated against the tool's declared output schema.

Steps are described with :class:`ComposedStep`. Each step references an
existing registered tool by name and supplies an ``args_builder`` callable
that builds the per-step argument dict from the composed tool's own
parsed input plus the list of prior step outputs. Failure of any inner
step propagates out of :meth:`ComposedTool.invoke` as a
``ToolInvocationError`` carrying the failing step index, tool name, and
the underlying error message.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from agentic_runner.tools._base import (
    REGISTRY,
    Tool,
    ToolInvocationError,
    register_tool,
)


@dataclass(frozen=True)
class ComposedStep:
    """One inner step of a composed tool.

    ``tool_name`` must reference an already-registered tool. ``args_builder``
    receives the composed tool's parsed input dict and the list of prior
    step output dicts (in execution order) and returns the kwargs to pass
    to the inner tool.
    """

    tool_name: str
    args_builder: Callable[[dict[str, Any], list[dict[str, Any]]], dict[str, Any]]


class ComposedTool:
    """A composed tool — fixed sequence of inner tool calls as one primitive.

    Subclasses declare the standard Tool surface (``name``, ``description``,
    ``input_model``, ``output_model``, ``max_runtime_ms``, ``idempotent``)
    and the ``steps`` and ``output_combiner`` class attributes that drive
    the inner sequence.
    """

    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    max_runtime_ms: int = 10_000
    idempotent: bool = True

    steps: list[ComposedStep]
    output_combiner: Callable[[dict[str, Any], list[dict[str, Any]]], dict[str, Any]]

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        parsed = self.input_model.model_validate(args).model_dump()
        prior: list[dict[str, Any]] = []
        for idx, step in enumerate(self.steps):
            inner = REGISTRY.get(step.tool_name)
            if inner is None:
                raise ToolInvocationError(
                    f"{self.name}: step {idx} references unregistered tool " f"{step.tool_name!r}"
                )
            try:
                step_args = step.args_builder(parsed, prior)
            except Exception as exc:  # noqa: BLE001
                raise ToolInvocationError(
                    f"{self.name}: step {idx} ({step.tool_name}) " f"args_builder failed: {exc}"
                ) from exc
            try:
                # Validate per-inner-tool input model — surface schema
                # mismatches early with a precise message.
                inner.input_model.model_validate(step_args)
                output = inner.invoke(step_args)
            except ToolInvocationError as exc:
                raise ToolInvocationError(
                    f"{self.name}: step {idx} ({step.tool_name}) failed: {exc}"
                ) from exc
            except Exception as exc:  # noqa: BLE001
                raise ToolInvocationError(
                    f"{self.name}: step {idx} ({step.tool_name}) raised: {exc}"
                ) from exc
            prior.append(output)
        try:
            combined = self.output_combiner(parsed, prior)
        except Exception as exc:  # noqa: BLE001
            raise ToolInvocationError(f"{self.name}: output_combiner failed: {exc}") from exc
        # Validate the combined payload against the composed tool's own
        # output schema before returning.
        self.output_model.model_validate(combined)
        return combined


# ---------------------------------------------------------------------------
# Sample composed tool — `summarize_document`
#
#     read_file -> summarize -> finish
#
# Demonstrates the pattern: a single logical action with one input
# (``path`` + ``max_words``) and one output (``summary`` + ``source_path``
# + ``source_bytes``). The composed tool is registered under its own name
# and is selectable just like any other Tool.
# ---------------------------------------------------------------------------


from pydantic import Field  # noqa: E402  (kept here so the file reads top-to-bottom)


class SummarizeDocumentInput(BaseModel):
    path: str = Field(min_length=1, max_length=512)
    max_words: int = Field(default=20, ge=1, le=200)


class SummarizeDocumentOutput(BaseModel):
    summary: str
    source_path: str
    source_bytes: int
    word_count: int


def _read_args(parsed: dict[str, Any], prior: list[dict[str, Any]]) -> dict[str, Any]:
    return {"path": parsed["path"]}


def _summarize_args(parsed: dict[str, Any], prior: list[dict[str, Any]]) -> dict[str, Any]:
    return {"text": prior[0]["content"], "max_words": parsed["max_words"]}


def _finish_args(parsed: dict[str, Any], prior: list[dict[str, Any]]) -> dict[str, Any]:
    return {"result": prior[1]["summary"]}


def _combine(parsed: dict[str, Any], prior: list[dict[str, Any]]) -> dict[str, Any]:
    read_out, sum_out, _finish_out = prior
    return {
        "summary": sum_out["summary"],
        "source_path": parsed["path"],
        "source_bytes": read_out["bytes"],
        "word_count": sum_out["word_count"],
    }


@register_tool
class SummarizeDocumentTool(ComposedTool):
    name = "summarize_document"
    description = (
        "Composed tool: read_file -> summarize -> finish. Reads a workspace "
        "document and returns its summary plus source metadata as one "
        "atomic action."
    )
    input_model = SummarizeDocumentInput
    output_model = SummarizeDocumentOutput
    max_runtime_ms = 6_000
    idempotent = True

    steps: list[ComposedStep] = [
        ComposedStep(tool_name="read_file", args_builder=_read_args),
        ComposedStep(tool_name="summarize", args_builder=_summarize_args),
        ComposedStep(tool_name="finish", args_builder=_finish_args),
    ]
    output_combiner: Callable[[dict[str, Any], list[dict[str, Any]]], dict[str, Any]] = (
        staticmethod(_combine)
    )


# Sanity check at import time — the composed tool must satisfy the Tool protocol.
_inst = REGISTRY.get("summarize_document")
if _inst is None or not isinstance(_inst, Tool):  # pragma: no cover - import-time guard
    raise RuntimeError("summarize_document failed to register as a Tool")
