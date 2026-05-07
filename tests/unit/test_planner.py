"""Planner tests — JSON parsing, malformed plan rejection, replan reason flow."""

from __future__ import annotations

import json

import pytest

from agentic_runner.failure import FailureKind, FailureReason
from agentic_runner.planner import Planner, PlannerError
from agentic_runner.providers.base import ChatMessage, ChatResponse


class _StubProvider:
    name = "stub"

    def __init__(self, responses: list[ChatResponse]) -> None:
        self._responses = list(responses)
        self.last_messages: list[ChatMessage] | None = None

    def chat(self, messages, tools=None, model=None) -> ChatResponse:
        self.last_messages = list(messages)
        return self._responses.pop(0)


def test_planner_parses_json() -> None:
    resp = ChatResponse(
        text=json.dumps({"subtasks": [{"description": "do it", "tool_hint": "calculate"}]})
    )
    p = Planner(_StubProvider([resp]))
    plan, cost = p.plan("compute 1+1")
    assert plan.subtasks[0].tool_hint == "calculate"
    assert cost == 0.0


def test_planner_rejects_non_json() -> None:
    resp = ChatResponse(text="not json at all")
    p = Planner(_StubProvider([resp]))
    with pytest.raises(PlannerError):
        p.plan("anything")


def test_planner_rejects_missing_required_field() -> None:
    resp = ChatResponse(text=json.dumps({"subtasks": [{"description": "x"}]}))
    p = Planner(_StubProvider([resp]))
    with pytest.raises(PlannerError):
        p.plan("anything")


def test_planner_replan_includes_reason_in_prompt() -> None:
    resp = ChatResponse(text=json.dumps({"subtasks": []}))
    stub = _StubProvider([resp])
    p = Planner(stub)
    failure = FailureReason(kind=FailureKind.OUTPUT_SCHEMA_MISMATCH, message="bad output")
    p.plan("retry", failure_reason=failure)
    assert stub.last_messages is not None
    plan_msg = stub.last_messages[-1].content
    assert "REPLAN_REASON" in plan_msg
    assert "output_schema_mismatch" in plan_msg
