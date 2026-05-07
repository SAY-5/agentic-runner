"""Runner — the plan-validate-replan loop.

State machine summary::

    PLAN ──► [for subtask in plan]
              │
              ├─► SELECT tool
              │       │
              │       ├─► (no tool) ────────► REPLAN with NO_TOOL_FOR_SUBTASK
              │       │
              │       └─► INVOKE tool ──► VALIDATE
              │                              │
              │                              ├─► valid ─► next subtask
              │                              │
              │                              └─► invalid ─► REPLAN with FailureReason
              │
              └─► finish tool reached ─► SUCCEEDED
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from agentic_runner import models
from agentic_runner.failure import FailureKind, FailureReason
from agentic_runner.models import (
    Goal,
    GoalStatus,
    ReplanEvent,
    Subtask,
    SubtaskStatus,
    ToolCall,
    ToolCallStatus,
    ValidationResult,
)
from agentic_runner.planner import Planner, PlannerError
from agentic_runner.providers.base import ChatProvider
from agentic_runner.selector import Selector
from agentic_runner.tools import REGISTRY
from agentic_runner.tools._base import (
    ToolInvocationError,
    ToolTimeoutError,
    invoke_with_guard,
)
from agentic_runner.trace import get_logger, span
from agentic_runner.validator import Validator

_log = get_logger("runner")


@dataclass
class RunBudget:
    max_steps: int = 20
    max_replans: int = 5
    max_cost_usd: float = 0.50
    max_wall_clock_s: float = 60.0


@dataclass
class RunResult:
    status: GoalStatus
    final_result: str | None = None
    abort_reason: str | None = None
    total_steps: int = 0
    replan_count: int = 0
    total_cost_usd: float = 0.0
    trace: list[dict[str, Any]] = field(default_factory=list)


class Runner:
    """Orchestrates the loop for a single goal."""

    def __init__(
        self,
        provider: ChatProvider,
        session_factory: Any,
        budget: RunBudget | None = None,
    ) -> None:
        self._provider = provider
        self._session_factory = session_factory
        self._budget = budget or RunBudget()
        self._planner = Planner(provider)
        self._selector = Selector(provider)
        self._validator = Validator()

    def run(self, goal_text: str, goal_id: str | None = None) -> RunResult:
        started_at = time.perf_counter()
        session: Session = self._session_factory()
        try:
            goal = self._init_goal(session, goal_text, goal_id)
            return self._run_loop(session, goal, started_at)
        finally:
            session.close()

    def _init_goal(self, session: Session, goal_text: str, goal_id: str | None) -> Goal:
        if goal_id is not None:
            goal = session.get(Goal, goal_id)
            if goal is None:
                raise ValueError(f"goal not found: {goal_id}")
        else:
            goal = Goal(
                goal_text=goal_text,
                max_steps=self._budget.max_steps,
                max_replans=self._budget.max_replans,
                max_cost_usd=self._budget.max_cost_usd,
            )
            session.add(goal)
            session.commit()
            session.refresh(goal)
        goal.status = GoalStatus.RUNNING
        session.commit()
        return goal

    def _run_loop(self, session: Session, goal: Goal, started_at: float) -> RunResult:
        result = RunResult(status=GoalStatus.RUNNING)
        steps_taken = 0
        replan_count = 0
        cost_usd = 0.0
        last_failure: FailureReason | None = None
        state: dict[str, Any] = {"prior_outputs": []}

        while True:
            elapsed = time.perf_counter() - started_at
            if elapsed > self._budget.max_wall_clock_s:
                return self._abort(
                    session,
                    goal,
                    result,
                    "wall_clock_exceeded",
                    steps_taken,
                    replan_count,
                    cost_usd,
                )
            if replan_count > self._budget.max_replans:
                return self._abort(
                    session,
                    goal,
                    result,
                    f"max_replans_exceeded ({replan_count})",
                    steps_taken,
                    replan_count,
                    cost_usd,
                )

            try:
                with span("planner.plan", replan=replan_count):
                    plan, plan_cost = self._planner.plan(
                        goal.goal_text, state=state, failure_reason=last_failure
                    )
                cost_usd += plan_cost
            except PlannerError as exc:
                return self._abort(
                    session,
                    goal,
                    result,
                    f"planner_error: {exc}",
                    steps_taken,
                    replan_count,
                    cost_usd,
                )

            if not plan.subtasks:
                return self._abort(
                    session,
                    goal,
                    result,
                    "planner_returned_empty_plan",
                    steps_taken,
                    replan_count,
                    cost_usd,
                )

            if cost_usd > self._budget.max_cost_usd:
                return self._abort(
                    session,
                    goal,
                    result,
                    f"max_cost_exceeded ({cost_usd:.4f} > {self._budget.max_cost_usd})",
                    steps_taken,
                    replan_count,
                    cost_usd,
                )

            base_idx = len(goal.subtasks)
            db_subtasks: list[Subtask] = []
            for i, ps in enumerate(plan.subtasks):
                st = Subtask(
                    goal_id=goal.id,
                    idx=base_idx + i,
                    description=ps.description,
                    confidence_threshold=ps.confidence_threshold,
                )
                session.add(st)
                db_subtasks.append(st)
            session.commit()

            need_replan = False
            for ps, db_st in zip(plan.subtasks, db_subtasks, strict=True):
                if steps_taken >= self._budget.max_steps:
                    return self._abort(
                        session,
                        goal,
                        result,
                        f"max_steps_exceeded ({steps_taken})",
                        steps_taken,
                        replan_count,
                        cost_usd,
                    )
                if cost_usd > self._budget.max_cost_usd:
                    return self._abort(
                        session,
                        goal,
                        result,
                        f"max_cost_exceeded ({cost_usd:.4f} > {self._budget.max_cost_usd})",
                        steps_taken,
                        replan_count,
                        cost_usd,
                    )

                with span("selector.choose_tool", subtask_idx=db_st.idx):
                    sel = self._selector.choose_tool(
                        goal.goal_text,
                        ps.description,
                        ps.tool_hint,
                        args_override=ps.args_override,
                    )
                cost_usd += sel.cost_usd

                if sel.failure is not None or sel.tool_call is None:
                    failure = sel.failure or FailureReason(
                        kind=FailureKind.NO_TOOL_FOR_SUBTASK,
                        message="selector returned no tool_call",
                    )
                    db_st.status = SubtaskStatus.FAILED
                    session.commit()
                    last_failure = failure
                    if failure.kind == FailureKind.NO_TOOL_FOR_SUBTASK:
                        replan_count += 1
                        self._record_replan(session, goal, replan_count, None, failure, "")
                        if replan_count > self._budget.max_replans:
                            return self._abort(
                                session,
                                goal,
                                result,
                                f"max_replans_exceeded :: {failure.short()}",
                                steps_taken,
                                replan_count,
                                cost_usd,
                            )
                    need_replan = True
                    break

                args = dict(sel.tool_call.arguments)
                if args.get("text") == "__USE_PRIOR_OUTPUT__" and state["prior_outputs"]:
                    last_out = state["prior_outputs"][-1]
                    args["text"] = (
                        last_out.get("content") or last_out.get("summary") or str(last_out)
                    )

                tool = REGISTRY[sel.tool_call.tool_name]
                tc = ToolCall(
                    subtask_id=db_st.id,
                    idx=len(db_st.tool_calls),
                    tool_name=tool.name,
                    args=args,
                )
                session.add(tc)
                session.commit()
                steps_taken += 1
                db_st.status = SubtaskStatus.RUNNING

                try:
                    output, latency_ms = invoke_with_guard(tool, args)
                    tc.output = output
                    tc.latency_ms = latency_ms
                    tc.status = ToolCallStatus.OK
                except ToolTimeoutError as exc:
                    tc.status = ToolCallStatus.TIMEOUT
                    tc.output = {"error": str(exc)}
                    db_st.status = SubtaskStatus.FAILED
                    session.commit()
                    last_failure = FailureReason(
                        kind=FailureKind.TIMEOUT, message=str(exc), tool_name=tool.name
                    )
                    replan_count += 1
                    self._record_replan(session, goal, replan_count, tc.id, last_failure, "")
                    need_replan = True
                    break
                except ToolInvocationError as exc:
                    tc.status = ToolCallStatus.ERROR
                    tc.output = {"error": str(exc)}
                    db_st.status = SubtaskStatus.FAILED
                    session.commit()
                    last_failure = FailureReason(
                        kind=FailureKind.TOOL_RETURNED_ERROR,
                        message=str(exc),
                        tool_name=tool.name,
                    )
                    replan_count += 1
                    self._record_replan(session, goal, replan_count, tc.id, last_failure, "")
                    need_replan = True
                    break

                outcome = self._validator.validate(
                    tool.name,
                    output,
                    tool.output_model,
                    confidence_threshold=ps.confidence_threshold,
                )
                v = ValidationResult(
                    tool_call_id=tc.id,
                    valid=outcome.valid,
                    failure_reason=(outcome.failure.kind.value if outcome.failure else None),
                    schema_violations=(
                        outcome.failure.schema_violations if outcome.failure else None
                    ),
                )
                session.add(v)
                session.commit()

                if not outcome.valid:
                    db_st.status = SubtaskStatus.FAILED
                    session.commit()
                    assert outcome.failure is not None
                    last_failure = outcome.failure
                    replan_count += 1
                    self._record_replan(session, goal, replan_count, tc.id, last_failure, "")
                    need_replan = True
                    break

                db_st.status = SubtaskStatus.DONE
                state["prior_outputs"].append(output)
                session.commit()

                if tool.name == "finish":
                    final = output["final"]
                    return self._succeed(
                        session, goal, result, final, steps_taken, replan_count, cost_usd
                    )

            if not need_replan:
                return self._abort(
                    session,
                    goal,
                    result,
                    "plan_exhausted_without_finish",
                    steps_taken,
                    replan_count,
                    cost_usd,
                )

    def _succeed(
        self,
        session: Session,
        goal: Goal,
        result: RunResult,
        final_text: str,
        steps_taken: int,
        replan_count: int,
        cost_usd: float,
    ) -> RunResult:
        goal.status = GoalStatus.SUCCEEDED
        goal.final_result = final_text
        goal.total_steps = steps_taken
        goal.replan_count = replan_count
        goal.total_cost_usd = cost_usd
        session.commit()
        result.status = GoalStatus.SUCCEEDED
        result.final_result = final_text
        result.total_steps = steps_taken
        result.replan_count = replan_count
        result.total_cost_usd = cost_usd
        result.trace = _trace_for_goal(session, goal.id)
        _log.info(
            "goal_succeeded",
            goal_id=goal.id,
            steps=steps_taken,
            replans=replan_count,
            cost=cost_usd,
        )
        return result

    def _abort(
        self,
        session: Session,
        goal: Goal,
        result: RunResult,
        reason: str,
        steps_taken: int,
        replan_count: int,
        cost_usd: float,
    ) -> RunResult:
        goal.status = GoalStatus.ABORTED
        goal.abort_reason = reason
        goal.total_steps = steps_taken
        goal.replan_count = replan_count
        goal.total_cost_usd = cost_usd
        session.commit()
        result.status = GoalStatus.ABORTED
        result.abort_reason = reason
        result.total_steps = steps_taken
        result.replan_count = replan_count
        result.total_cost_usd = cost_usd
        result.trace = _trace_for_goal(session, goal.id)
        _log.info(
            "goal_aborted",
            goal_id=goal.id,
            reason=reason,
            steps=steps_taken,
            replans=replan_count,
            cost=cost_usd,
        )
        return result

    def _record_replan(
        self,
        session: Session,
        goal: Goal,
        idx: int,
        tool_call_id: str | None,
        failure: FailureReason,
        summary: str,
    ) -> None:
        rp = ReplanEvent(
            goal_id=goal.id,
            idx=idx,
            triggered_by_tool_call_id=tool_call_id,
            failure_reason=failure.kind.value,
            new_plan_summary=summary,
        )
        session.add(rp)
        session.commit()


def _trace_for_goal(session: Session, goal_id: str) -> list[dict[str, Any]]:
    """Materialize the trace tree."""
    goal = session.get(models.Goal, goal_id)
    if goal is None:
        return []
    subtasks: list[dict[str, Any]] = []
    for st in goal.subtasks:
        tcs: list[dict[str, Any]] = []
        for tc in st.tool_calls:
            tcs.append(
                {
                    "idx": tc.idx,
                    "tool": tc.tool_name,
                    "args": tc.args,
                    "output": tc.output,
                    "status": tc.status.value,
                    "latency_ms": tc.latency_ms,
                    "validation": (
                        {
                            "valid": tc.validation.valid,
                            "failure_reason": tc.validation.failure_reason,
                            "schema_violations": tc.validation.schema_violations,
                        }
                        if tc.validation
                        else None
                    ),
                }
            )
        subtasks.append(
            {
                "idx": st.idx,
                "description": st.description,
                "status": st.status.value,
                "tool_calls": tcs,
            }
        )
    return [
        {
            "subtasks": subtasks,
            "replan_events": [
                {
                    "idx": rp.idx,
                    "failure_reason": rp.failure_reason,
                    "triggered_by_tool_call_id": rp.triggered_by_tool_call_id,
                }
                for rp in goal.replan_events
            ],
        }
    ]
