"""Parallel subtask scheduler — DAG-driven async execution.

The sequential :class:`agentic_runner.runner.Runner` walks a plan one
subtask at a time. ``run_parallel`` keeps the same plan-validate-replan
shape but schedules subtasks against an explicit dependency DAG: any
subtask whose dependencies are satisfied runs as soon as a worker slot is
free, capped at ``MAX_CONCURRENCY`` simultaneous tasks via
:class:`asyncio.Semaphore`.

Plan -> DAG mapping:

* ``PlannedSubtask.dependencies`` is the load-bearing field. It is a list
  of indices (within the same plan) that must complete before the
  subtask is eligible to run.
* If ``dependencies`` is empty AND ``parallel`` is true, the subtask is
  treated as a root with no implicit dependency on prior siblings.
* If ``dependencies`` is empty AND ``parallel`` is false, the subtask
  inherits an implicit dependency on every earlier subtask, preserving
  the sequential semantics for plans that don't opt in.

Failure-replan semantics: when any subtask fails, the scheduler waits
for in-flight tasks to settle (``return_exceptions=True``), captures the
state from every completed subtask (success or skipped), and bubbles a
typed :class:`agentic_runner.failure.FailureReason` up to the runner so
the planner can replan with the latest sibling state.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from agentic_runner.failure import FailureKind, FailureReason
from agentic_runner.models import GoalStatus
from agentic_runner.planner import Plan, PlannedSubtask
from agentic_runner.selector import Selector
from agentic_runner.tools import REGISTRY
from agentic_runner.tools._base import (
    Tool,
    ToolInvocationError,
    ToolTimeoutError,
    invoke_with_guard,
)
from agentic_runner.validator import Validator

MAX_CONCURRENCY: int = 4


@dataclass
class SubtaskOutcome:
    idx: int
    description: str
    tool_name: str | None
    output: dict[str, Any] | None
    started_at: float
    finished_at: float
    status: str  # "ok" | "failed" | "skipped"
    failure: FailureReason | None = None


@dataclass
class ParallelResult:
    status: GoalStatus
    final_result: str | None
    abort_reason: str | None
    completed: list[SubtaskOutcome] = field(default_factory=list)
    duration_s: float = 0.0
    last_failure: FailureReason | None = None


def _build_dependency_map(subtasks: list[PlannedSubtask]) -> dict[int, set[int]]:
    """Translate ``PlannedSubtask.dependencies`` into a complete DAG.

    Sequential subtasks with no explicit dependencies are given an
    implicit dependency on every prior subtask, preserving the
    legacy ordering. Parallel subtasks with no dependencies are
    treated as roots.
    """
    dep: dict[int, set[int]] = {}
    for i, ps in enumerate(subtasks):
        explicit = set(ps.dependencies)
        if not explicit and not ps.parallel:
            explicit = set(range(i))
        # Validate: dependencies must point to earlier subtasks only.
        for d in list(explicit):
            if d >= i or d < 0:
                explicit.discard(d)
        dep[i] = explicit
    return dep


def _ready(deps: dict[int, set[int]], done: set[int], scheduled: set[int]) -> list[int]:
    return [i for i, d in deps.items() if i not in done and i not in scheduled and d <= done]


async def _run_one_subtask(
    idx: int,
    ps: PlannedSubtask,
    selector: Selector,
    validator: Validator,
    goal_text: str,
    sem: asyncio.Semaphore,
) -> SubtaskOutcome:
    """Select + invoke + validate a single subtask. Bubbles failures
    back to the scheduler via ``SubtaskOutcome.status``."""
    started = time.perf_counter()
    async with sem:

        def _select_and_invoke() -> tuple[str | None, dict[str, Any] | None, FailureReason | None]:
            sel = selector.choose_tool(
                goal_text,
                ps.description,
                ps.tool_hint,
                args_override=ps.args_override,
            )
            if sel.failure is not None or sel.tool_call is None:
                fail = sel.failure or FailureReason(
                    kind=FailureKind.NO_TOOL_FOR_SUBTASK,
                    message="selector returned no tool_call",
                )
                return None, None, fail

            args = dict(sel.tool_call.arguments)
            tool: Tool = REGISTRY[sel.tool_call.tool_name]
            try:
                output, _ = invoke_with_guard(tool, args)
            except ToolTimeoutError as exc:
                return (
                    tool.name,
                    None,
                    FailureReason(kind=FailureKind.TIMEOUT, message=str(exc), tool_name=tool.name),
                )
            except ToolInvocationError as exc:
                return (
                    tool.name,
                    None,
                    FailureReason(
                        kind=FailureKind.TOOL_RETURNED_ERROR,
                        message=str(exc),
                        tool_name=tool.name,
                    ),
                )

            outcome = validator.validate(
                tool.name,
                output,
                tool.output_model,
                confidence_threshold=ps.confidence_threshold,
            )
            if not outcome.valid:
                return tool.name, output, outcome.failure
            return tool.name, output, None

        tool_name, output, failure = await asyncio.to_thread(_select_and_invoke)

    finished = time.perf_counter()
    if failure is not None:
        return SubtaskOutcome(
            idx=idx,
            description=ps.description,
            tool_name=tool_name,
            output=output,
            started_at=started,
            finished_at=finished,
            status="failed",
            failure=failure,
        )
    return SubtaskOutcome(
        idx=idx,
        description=ps.description,
        tool_name=tool_name,
        output=output,
        started_at=started,
        finished_at=finished,
        status="ok",
    )


async def run_parallel(
    plan: Plan,
    goal_text: str,
    selector: Selector,
    validator: Validator,
    max_concurrency: int = MAX_CONCURRENCY,
) -> ParallelResult:
    """Execute a plan against the explicit dependency DAG.

    The result reports every subtask that completed (successfully or
    skipped due to a sibling failure) so the caller can hand the latest
    state to the planner for replan.
    """
    started_at = time.perf_counter()
    deps = _build_dependency_map(plan.subtasks)
    sem = asyncio.Semaphore(max_concurrency)

    completed: dict[int, SubtaskOutcome] = {}
    scheduled: set[int] = set()
    in_flight: dict[int, asyncio.Task[SubtaskOutcome]] = {}
    last_failure: FailureReason | None = None
    aborted = False

    while True:
        # Schedule every subtask whose dependencies are now satisfied.
        for idx in _ready(deps, set(completed.keys()), scheduled):
            ps = plan.subtasks[idx]
            scheduled.add(idx)
            in_flight[idx] = asyncio.create_task(
                _run_one_subtask(idx, ps, selector, validator, goal_text, sem)
            )

        if not in_flight:
            break

        done, _ = await asyncio.wait(in_flight.values(), return_when=asyncio.FIRST_COMPLETED)
        finished_indices: list[int] = []
        for task in done:
            outcome = task.result()
            completed[outcome.idx] = outcome
            finished_indices.append(outcome.idx)
            if outcome.status == "failed" and last_failure is None:
                last_failure = outcome.failure

        for idx in finished_indices:
            in_flight.pop(idx, None)

        if last_failure is not None:
            # Drain remaining in-flight tasks; do not schedule new ones.
            if in_flight:
                drained, _ = await asyncio.wait(
                    in_flight.values(), return_when=asyncio.ALL_COMPLETED
                )
                for task in drained:
                    outcome = task.result()
                    completed[outcome.idx] = outcome
                in_flight.clear()
            aborted = True
            break

    duration = time.perf_counter() - started_at

    if aborted:
        return ParallelResult(
            status=GoalStatus.RUNNING,  # caller decides abort vs replan
            final_result=None,
            abort_reason=None,
            completed=[completed[i] for i in sorted(completed)],
            duration_s=duration,
            last_failure=last_failure,
        )

    # Look for a finish-shaped outcome — by convention the `finish` tool
    # (or any tool whose output ships a `final` string) terminates the
    # plan. Prefer the highest-index finishing subtask so plans that
    # have multiple finish-shaped tools resolve to the last one.
    final_text: str | None = None
    for idx in sorted(completed.keys(), reverse=True):
        o = completed[idx]
        if (
            o.status == "ok"
            and o.output is not None
            and isinstance(o.output.get("final"), str)
            and (o.tool_name == "finish" or idx == max(completed.keys()))
        ):
            final_text = o.output["final"]
            break

    return ParallelResult(
        status=GoalStatus.SUCCEEDED if final_text is not None else GoalStatus.RUNNING,
        final_result=final_text,
        abort_reason=None,
        completed=[completed[i] for i in sorted(completed)],
        duration_s=duration,
    )
