"""Tests for the parallel subtask scheduler.

These tests use a stubbed Selector + Validator to drive the scheduler
directly. The runner-level integration test exercises ``Runner.run_parallel``
end-to-end.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from agentic_runner.failure import FailureKind, FailureReason
from agentic_runner.models import GoalStatus
from agentic_runner.parallel_runner import (
    MAX_CONCURRENCY,
    _build_dependency_map,
    run_parallel,
)
from agentic_runner.planner import Plan, PlannedSubtask
from agentic_runner.providers.fake import FakeProvider
from agentic_runner.selector import SelectorResult, ToolCallPlan
from agentic_runner.validator import Validator

# ---------------------------------------------------------------------------
# Helpers — a stubbed Selector that picks a fixed tool with fixed args.
# ---------------------------------------------------------------------------


class _StubSelector:
    """Selector double — yields a deterministic ToolCallPlan per subtask."""

    def __init__(self, mapping: dict[str, dict[str, Any]]) -> None:
        # Maps subtask description -> (tool_name, args).
        self._mapping = mapping

    def choose_tool(
        self,
        goal: str,
        subtask_description: str,
        tool_hint: str,
        args_override: dict[str, Any] | None = None,
    ) -> SelectorResult:
        spec = self._mapping.get(subtask_description)
        if spec is None:
            return SelectorResult(
                failure=FailureReason(
                    kind=FailureKind.NO_TOOL_FOR_SUBTASK,
                    message=f"no stub for {subtask_description!r}",
                )
            )
        return SelectorResult(
            tool_call=ToolCallPlan(tool_name=spec["tool_name"], arguments=dict(spec["args"])),
            cost_usd=0.0,
        )


# ---------------------------------------------------------------------------
# Dependency map construction
# ---------------------------------------------------------------------------


def test_dependency_map_implicit_sequential() -> None:
    plan = [
        PlannedSubtask(description="a", tool_hint="finish"),
        PlannedSubtask(description="b", tool_hint="finish"),
        PlannedSubtask(description="c", tool_hint="finish"),
    ]
    deps = _build_dependency_map(plan)
    # Sequential default: each waits for all earlier siblings.
    assert deps[0] == set()
    assert deps[1] == {0}
    assert deps[2] == {0, 1}


def test_dependency_map_parallel_roots() -> None:
    plan = [
        PlannedSubtask(description="a", tool_hint="finish", parallel=True),
        PlannedSubtask(description="b", tool_hint="finish", parallel=True),
        PlannedSubtask(description="c", tool_hint="finish", dependencies=[0, 1]),
    ]
    deps = _build_dependency_map(plan)
    assert deps[0] == set()
    assert deps[1] == set()
    assert deps[2] == {0, 1}


def test_dependency_map_drops_invalid_indices() -> None:
    plan = [
        PlannedSubtask(description="a", tool_hint="finish"),
        # forward reference (5 is out of range) gets dropped.
        PlannedSubtask(description="b", tool_hint="finish", dependencies=[5]),
    ]
    deps = _build_dependency_map(plan)
    assert deps[1] == set()


# ---------------------------------------------------------------------------
# Concurrency: independent parallel subtasks should overlap in wall time.
# ---------------------------------------------------------------------------


class _SlowFinishTool:
    """Inline tool registered just for timing tests."""

    name = "slow_finish"
    description = "Sleeps then returns final."

    def __init__(self) -> None:
        from pydantic import BaseModel, Field

        class _In(BaseModel):
            result: str = Field(min_length=1)
            sleep_s: float = Field(default=0.05, ge=0.0, le=2.0)

        class _Out(BaseModel):
            final: str

        self.input_model = _In
        self.output_model = _Out
        self.max_runtime_ms = 5_000
        self.idempotent = True

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        time.sleep(args.get("sleep_s", 0.05))
        return {"final": args["result"]}


@pytest.fixture()
def _slow_tool_registered() -> Any:
    from agentic_runner.tools._base import REGISTRY

    inst = _SlowFinishTool()
    REGISTRY[inst.name] = inst
    yield inst
    REGISTRY.pop(inst.name, None)


def test_parallel_subtasks_overlap_in_wall_time(_slow_tool_registered: Any) -> None:
    """5-subtask plan with 2 independent parallel + 3 sequential — assert
    the parallel pair's wall time is meaningfully shorter than serial."""
    sleep_s = 0.20
    plan = Plan(
        subtasks=[
            PlannedSubtask(description="par_a", tool_hint="slow_finish", parallel=True),
            PlannedSubtask(description="par_b", tool_hint="slow_finish", parallel=True),
            PlannedSubtask(description="seq_c", tool_hint="slow_finish", dependencies=[0, 1]),
            PlannedSubtask(description="seq_d", tool_hint="slow_finish", dependencies=[2]),
            PlannedSubtask(description="finish_e", tool_hint="slow_finish", dependencies=[3]),
        ]
    )
    selector = _StubSelector(
        {
            "par_a": {"tool_name": "slow_finish", "args": {"result": "A", "sleep_s": sleep_s}},
            "par_b": {"tool_name": "slow_finish", "args": {"result": "B", "sleep_s": sleep_s}},
            "seq_c": {"tool_name": "slow_finish", "args": {"result": "C", "sleep_s": sleep_s}},
            "seq_d": {"tool_name": "slow_finish", "args": {"result": "D", "sleep_s": sleep_s}},
            "finish_e": {
                "tool_name": "slow_finish",
                "args": {"result": "E", "sleep_s": sleep_s},
            },
        }
    )
    validator = Validator()

    result = asyncio.run(
        run_parallel(plan, "g", selector, validator, max_concurrency=4)  # type: ignore[arg-type]
    )

    assert result.status == GoalStatus.SUCCEEDED
    assert result.final_result == "E"
    assert len(result.completed) == 5

    # Sequential lower bound is 5 * sleep_s. Parallel should save ~one
    # sleep_s by overlapping par_a + par_b. Allow generous slack.
    sequential_floor = 5 * sleep_s
    upper = sequential_floor - sleep_s * 0.5
    assert result.duration_s < upper, (
        f"parallel wall {result.duration_s:.3f}s did not beat "
        f"{upper:.3f}s threshold (5*sleep={sequential_floor:.3f}s)"
    )

    # Verify par_a and par_b actually overlapped — their windows intersect.
    a = next(o for o in result.completed if o.idx == 0)
    b = next(o for o in result.completed if o.idx == 1)
    overlap = min(a.finished_at, b.finished_at) - max(a.started_at, b.started_at)
    assert overlap > 0, (
        f"par_a and par_b windows did not overlap "
        f"(a={a.started_at:.3f}-{a.finished_at:.3f}, "
        f"b={b.started_at:.3f}-{b.finished_at:.3f})"
    )


def test_parallel_result_matches_sequential(_slow_tool_registered: Any) -> None:
    """A plan run twice (parallel + serial-equivalent ordering) yields
    identical final results."""
    plan_par = Plan(
        subtasks=[
            PlannedSubtask(description="x1", tool_hint="slow_finish", parallel=True),
            PlannedSubtask(description="x2", tool_hint="slow_finish", parallel=True),
            PlannedSubtask(description="finish_z", tool_hint="slow_finish", dependencies=[0, 1]),
        ]
    )
    plan_seq = Plan(
        subtasks=[
            PlannedSubtask(description="x1", tool_hint="slow_finish"),
            PlannedSubtask(description="x2", tool_hint="slow_finish"),
            PlannedSubtask(description="finish_z", tool_hint="slow_finish"),
        ]
    )
    sel = _StubSelector(
        {
            "x1": {"tool_name": "slow_finish", "args": {"result": "X1", "sleep_s": 0.01}},
            "x2": {"tool_name": "slow_finish", "args": {"result": "X2", "sleep_s": 0.01}},
            "finish_z": {
                "tool_name": "slow_finish",
                "args": {"result": "Z", "sleep_s": 0.01},
            },
        }
    )
    val = Validator()
    par = asyncio.run(run_parallel(plan_par, "g", sel, val))  # type: ignore[arg-type]
    seq = asyncio.run(run_parallel(plan_seq, "g", sel, val, max_concurrency=1))  # type: ignore[arg-type]
    assert par.final_result == seq.final_result == "Z"


# ---------------------------------------------------------------------------
# Failure -> replan path
# ---------------------------------------------------------------------------


def test_failure_in_parallel_branch_surfaces_failure() -> None:
    """A failing parallel branch returns ``last_failure`` for replan."""
    plan = Plan(
        subtasks=[
            PlannedSubtask(description="bad", tool_hint="not_a_tool", parallel=True),
            PlannedSubtask(description="ok", tool_hint="finish", parallel=True),
        ]
    )
    sel = _StubSelector(
        {
            "ok": {"tool_name": "finish", "args": {"result": "ok"}},
            # 'bad' has no stub — selector returns NO_TOOL_FOR_SUBTASK.
        }
    )
    val = Validator()
    result = asyncio.run(run_parallel(plan, "g", sel, val))  # type: ignore[arg-type]
    assert result.last_failure is not None
    assert result.last_failure.kind == FailureKind.NO_TOOL_FOR_SUBTASK


def test_concurrency_cap_respected() -> None:
    assert MAX_CONCURRENCY == 4


# ---------------------------------------------------------------------------
# Runner.run_parallel — end-to-end smoke
# ---------------------------------------------------------------------------


def test_runner_run_parallel_succeeds_on_known_goal(db_session: Any) -> None:
    """Smoke test — Runner.run_parallel completes a FakeProvider goal."""
    from agentic_runner.runner import Runner

    runner = Runner(FakeProvider(), db_session)
    result = asyncio.run(
        runner.run_parallel("Compute (2+3)*7 with calculate")  # type: ignore[arg-type]
    )
    assert result.status == GoalStatus.SUCCEEDED
    assert "35" in (result.final_result or "")


def test_runner_run_parallel_replans_on_failure(db_session: Any) -> None:
    """Smoke test — Runner.run_parallel triggers replan + abort path
    correctly for the unknown-tool goal."""
    from agentic_runner.runner import RunBudget, Runner

    runner = Runner(FakeProvider(), db_session, budget=RunBudget(max_replans=1))
    result = asyncio.run(
        runner.run_parallel("Send an email to alice@example.com about the meeting")
    )
    assert result.status == GoalStatus.ABORTED
