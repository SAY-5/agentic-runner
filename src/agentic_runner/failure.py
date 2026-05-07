"""Typed failure reasons that drive the replan path."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class FailureKind(str, Enum):
    """Categorized reasons a step (or plan) can fail."""

    OUTPUT_SCHEMA_MISMATCH = "output_schema_mismatch"
    TOOL_RETURNED_ERROR = "tool_returned_error"
    CONFIDENCE_TOO_LOW = "confidence_too_low"
    PRECONDITION_VIOLATED = "precondition_violated"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    PLAN_INVALID = "plan_invalid"
    NO_TOOL_FOR_SUBTASK = "no_tool_for_subtask"
    TIMEOUT = "timeout"


class FailureReason(BaseModel):
    """Structured failure information that gets fed back into the planner."""

    kind: FailureKind
    message: str = ""
    tool_name: str | None = None
    schema_violations: list[dict[str, Any]] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)

    def short(self) -> str:
        return f"{self.kind.value}: {self.message}" if self.message else self.kind.value


class AbortGoal(BaseModel):
    """Honest termination signal — the planner has no path forward."""

    reason: str
    failure: FailureReason | None = None
