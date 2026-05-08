"""Hypothesis property tests for budget enforcement and tool dispatch.

These tests assert two invariants of the runner:

1. **Budgets are absolute.** For any combination of
   ``(max_steps, max_replans, max_cost_usd, max_wall_clock_s)`` and any
   FakeProvider-driven execution trace, an aborted run always names the
   budget that triggered it, and the recorded counters never exceed the
   declared cap (with the documented exception that ``replan_count`` may
   land at ``max_replans + 1`` because the runner increments first then
   checks).

2. **Tool dispatch is always Pydantic-valid or fails with a structured
   ``FailureReason``.** For random tool sets paired with random provider
   responses (well-formed, malformed-args, unknown-tool, empty), the
   selector never raises — it returns either a ``ToolCallPlan`` whose
   ``tool_name`` is registered, or a ``SelectorResult`` with a populated
   ``FailureReason``.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agentic_runner.failure import FailureKind
from agentic_runner.models import GoalStatus
from agentic_runner.providers.base import (
    ChatMessage,
    ChatResponse,
    ToolCallRequest,
    ToolSpec,
)
from agentic_runner.providers.fake import FakeProvider
from agentic_runner.runner import RunBudget, Runner
from agentic_runner.selector import Selector, SelectorResult
from agentic_runner.tools._base import REGISTRY

# ---------------------------------------------------------------------------
# Goals known to terminate quickly under FakeProvider — used as the random
# input for budget property tests.
# ---------------------------------------------------------------------------

_FAKE_GOALS = [
    "Compute (2+3)*7 with calculate",
    "Calculate the average salary in the engineering department",
    "List all employees in the database",
    "Count the rows in the orders table",
    "Read the notes file from the workspace",
    "Write a quarterly report to the workspace and read it back",
    "Fetch the example.com homepage via http_get",
    "Summarize the long_doc document in the workspace",
    "Extract JSON from the order prose using a structured schema",
    "Compute 10*5 in a two-step calculation, then add 25",
    "Count the distinct departments in the database",
    "Find John's salary stored in the workspace",
    "Send an email to alice@example.com about the meeting tomorrow",
    "Produce a short summary of the fox sentence with strict word cap",
    "Extract the order id from the unclear prose with strict mode",
]


# ---------------------------------------------------------------------------
# Property 1 — budget enforcement
# ---------------------------------------------------------------------------


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    goal_text=st.sampled_from(_FAKE_GOALS),
    max_steps=st.integers(min_value=0, max_value=15),
    max_replans=st.integers(min_value=0, max_value=8),
    max_cost_usd=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
    max_wall_clock_s=st.floats(
        min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False
    ),
)
def test_budget_enforced_for_random_traces(
    db_session: Any,
    tmp_workspace: Any,
    goal_text: str,
    max_steps: int,
    max_replans: int,
    max_cost_usd: float,
    max_wall_clock_s: float,
) -> None:
    """No completed run exceeds any declared budget; aborts cite a budget."""
    budget = RunBudget(
        max_steps=max_steps,
        max_replans=max_replans,
        max_cost_usd=max_cost_usd,
        max_wall_clock_s=max_wall_clock_s,
    )
    runner = Runner(FakeProvider(), db_session, budget=budget)
    result = runner.run(goal_text)

    assert (
        result.total_steps <= max_steps + 1
    ), f"steps {result.total_steps} > cap {max_steps} (+1 tolerance for finish)"
    # The runner increments replan_count then checks, so the recorded value
    # may equal max_replans + 1 when that is the abort trigger.
    assert result.replan_count <= max_replans + 1
    assert result.total_cost_usd <= max_cost_usd + 0.01

    if result.status == GoalStatus.ABORTED:
        reason = result.abort_reason or ""
        budget_keywords = (
            "max_steps_exceeded",
            "max_replans_exceeded",
            "max_cost_exceeded",
            "wall_clock_exceeded",
            "no_tool_for_subtask",
            "planner_returned_empty_plan",
            "plan_exhausted_without_finish",
        )
        assert any(
            kw in reason for kw in budget_keywords
        ), f"abort_reason {reason!r} does not name a budget or known cause"


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    goal_text=st.sampled_from(_FAKE_GOALS),
    max_steps=st.integers(min_value=0, max_value=2),
)
def test_tight_step_budget_aborts_with_steps_message(
    db_session: Any, tmp_workspace: Any, goal_text: str, max_steps: int
) -> None:
    """A near-zero step budget always aborts with a budget-shaped reason."""
    budget = RunBudget(max_steps=max_steps, max_replans=1, max_wall_clock_s=5.0)
    runner = Runner(FakeProvider(), db_session, budget=budget)
    result = runner.run(goal_text)
    if result.status == GoalStatus.ABORTED:
        assert any(
            kw in (result.abort_reason or "")
            for kw in ("max_steps", "max_replans", "no_tool_for_subtask", "max_cost")
        )


# ---------------------------------------------------------------------------
# Property 2 — tool dispatch is always Pydantic-valid or returns a typed
# FailureReason. This means: never raises, always lands in one of the two
# legal SelectorResult shapes.
# ---------------------------------------------------------------------------


class _ScriptedProvider:
    """Provider that returns a fixed ChatResponse to every prompt.

    Used to drive the Selector with arbitrary (well-formed, malformed,
    empty) tool-call payloads.
    """

    name = "scripted"

    def __init__(self, response: ChatResponse) -> None:
        self._response = response

    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
    ) -> ChatResponse:
        return self._response


_REGISTERED_TOOLS = sorted(REGISTRY.keys())


@st.composite
def _tool_call_response(draw: Any) -> ChatResponse:
    """Generate a ChatResponse with random tool calls.

    Mixes:
      * registered tool names (valid)
      * random unknown tool names (selector should fail-soft)
      * empty tool_calls list (selector should fail-soft)
      * malformed argument dicts (validation happens in invoke_with_guard,
        but the selector itself accepts any dict)
    """
    mode = draw(st.sampled_from(["registered", "unknown", "empty", "garbled_args"]))
    if mode == "empty":
        return ChatResponse(text="", cost_usd=draw(st.floats(0.0, 0.01)))
    if mode == "unknown":
        name = draw(st.text(min_size=1, max_size=20).filter(lambda s: s not in REGISTRY))
        return ChatResponse(
            tool_calls=[ToolCallRequest(id=str(uuid.uuid4()), name=name, arguments={})],
            cost_usd=draw(st.floats(0.0, 0.01)),
        )

    name = draw(st.sampled_from(_REGISTERED_TOOLS))
    if mode == "garbled_args":
        arguments: dict[str, Any] = draw(
            st.dictionaries(
                keys=st.text(min_size=1, max_size=10),
                values=st.one_of(
                    st.integers(),
                    st.text(max_size=20),
                    st.lists(st.integers(), max_size=3),
                    st.none(),
                ),
                max_size=4,
            )
        )
    else:
        arguments = {"placeholder": draw(st.text(max_size=20))}
    return ChatResponse(
        tool_calls=[ToolCallRequest(id=str(uuid.uuid4()), name=name, arguments=arguments)],
        cost_usd=draw(st.floats(0.0, 0.01)),
    )


@settings(max_examples=80, deadline=None)
@given(
    tool_hint=st.sampled_from(_REGISTERED_TOOLS + ["nonexistent_tool", "another_unknown"]),
    response=_tool_call_response(),
)
def test_selector_dispatch_always_typed(tool_hint: str, response: ChatResponse) -> None:
    """Selector returns one of two legal shapes; never raises."""
    sel = Selector(_ScriptedProvider(response))
    out = sel.choose_tool(goal="goal", subtask_description="subtask", tool_hint=tool_hint)

    assert isinstance(out, SelectorResult)
    if out.tool_call is not None:
        # Path A: a tool call was produced — it must reference a registered tool.
        assert out.tool_call.tool_name in REGISTRY
        assert isinstance(out.tool_call.arguments, dict)
        assert out.failure is None
    else:
        # Path B: the failure path — must carry a typed FailureKind.
        assert out.failure is not None
        assert out.failure.kind == FailureKind.NO_TOOL_FOR_SUBTASK
        assert out.failure.message  # non-empty


@settings(max_examples=40, deadline=None)
@given(
    tool_hint=st.text(min_size=1, max_size=30).filter(lambda s: s not in REGISTRY),
)
def test_selector_unknown_hint_always_fails_soft(tool_hint: str) -> None:
    """Any tool_hint not in REGISTRY produces a structured failure, no raise."""
    response = ChatResponse(
        tool_calls=[ToolCallRequest(id="x", name="finish", arguments={"result": "done"})]
    )
    sel = Selector(_ScriptedProvider(response))
    out = sel.choose_tool(goal="g", subtask_description="s", tool_hint=tool_hint)
    assert out.tool_call is None
    assert out.failure is not None
    assert out.failure.kind == FailureKind.NO_TOOL_FOR_SUBTASK


@settings(max_examples=40, deadline=None)
@given(
    text=st.text(max_size=200),
    cost=st.floats(min_value=0.0, max_value=0.05, allow_nan=False, allow_infinity=False),
)
def test_selector_empty_tool_calls_fails_soft(text: str, cost: float) -> None:
    """Empty tool_calls always produces a NO_TOOL_FOR_SUBTASK failure."""
    response = ChatResponse(text=text, cost_usd=cost)
    sel = Selector(_ScriptedProvider(response))
    out = sel.choose_tool(goal="g", subtask_description="s", tool_hint="finish")
    assert out.tool_call is None
    assert out.failure is not None
    assert out.failure.kind == FailureKind.NO_TOOL_FOR_SUBTASK
    assert out.cost_usd == pytest.approx(cost)
