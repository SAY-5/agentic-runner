"""Planner — converts a goal (and optional failure reason) into a list of subtasks."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from agentic_runner.failure import FailureKind, FailureReason
from agentic_runner.providers.base import ChatMessage, ChatProvider
from agentic_runner.tools._base import list_tool_specs
from agentic_runner.trace import get_logger

_log = get_logger("planner")


class PlannedSubtask(BaseModel):
    description: str = Field(min_length=1, max_length=500)
    tool_hint: str = Field(min_length=1, max_length=64)
    confidence_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    args_override: dict[str, Any] | None = None


class Plan(BaseModel):
    subtasks: list[PlannedSubtask]


class PlannerError(RuntimeError):
    """Raised when the planner produces an unparseable or empty plan."""


class Planner:
    """Builds a planning prompt and parses the JSON response."""

    def __init__(self, provider: ChatProvider) -> None:
        self._provider = provider

    def plan(
        self,
        goal: str,
        state: dict[str, Any] | None = None,
        failure_reason: FailureReason | None = None,
    ) -> tuple[Plan, float]:
        """Return a ``Plan`` plus the LLM cost incurred."""
        tools = list_tool_specs()
        tool_summary = ", ".join(t["name"] for t in tools)

        body: list[str] = [f"PLAN: {goal}", f"AVAILABLE_TOOLS: {tool_summary}"]
        if state:
            body.append(f"STATE: {json.dumps(state, default=str)[:1000]}")
        if failure_reason is not None:
            body.append(f"REPLAN_REASON: {failure_reason.kind.value} :: {failure_reason.message}")

        prompt = "\n".join(body)
        resp = self._provider.chat(
            [
                ChatMessage(role="user", content=goal),
                ChatMessage(role="user", content=prompt),
            ]
        )

        try:
            data = json.loads(resp.text) if resp.text else {"subtasks": []}
        except json.JSONDecodeError as exc:
            raise PlannerError(f"planner: response is not JSON: {exc}") from exc

        try:
            plan = Plan.model_validate(data)
        except Exception as exc:  # pragma: no cover - guarded by tests
            raise PlannerError(f"planner: response failed schema: {exc}") from exc

        if not plan.subtasks:
            _log.info("planner_returned_empty_plan", goal=goal[:80])

        return plan, resp.cost_usd

    @staticmethod
    def reason_for_invalid_plan(message: str) -> FailureReason:
        return FailureReason(kind=FailureKind.PLAN_INVALID, message=message)
