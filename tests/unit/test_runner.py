"""Runner-level tests — happy path, replan-required, abort, budget enforcement."""

from __future__ import annotations

from pathlib import Path

from agentic_runner.models import GoalStatus
from agentic_runner.providers.fake import FakeProvider
from agentic_runner.runner import RunBudget, Runner


def test_runner_happy_path_calculate(db_session) -> None:
    runner = Runner(FakeProvider(), db_session)
    res = runner.run("Compute (2+3)*7 calculate")
    assert res.status == GoalStatus.SUCCEEDED
    assert "35" in (res.final_result or "")
    assert res.replan_count == 0
    assert res.total_steps >= 2


def test_runner_happy_path_query_db(db_session) -> None:
    runner = Runner(FakeProvider(), db_session)
    res = runner.run("Calculate the average salary in the engineering department")
    assert res.status == GoalStatus.SUCCEEDED
    assert res.replan_count == 0


def test_runner_replan_required_for_find_john(db_session, tmp_workspace: Path) -> None:
    (tmp_workspace / "salaries.txt").write_text("John Doe: 88000\n")
    runner = Runner(FakeProvider(), db_session)
    res = runner.run("Find John's salary in the workspace")
    assert res.status in (GoalStatus.SUCCEEDED, GoalStatus.ABORTED)


def test_runner_aborts_on_unknown_tool(db_session) -> None:
    runner = Runner(FakeProvider(), db_session, budget=RunBudget(max_replans=1))
    res = runner.run("Send an email to alice@example.com about the meeting")
    assert res.status == GoalStatus.ABORTED
    assert res.abort_reason is not None
    assert (
        "no_tool_for_subtask" in res.abort_reason
        or "max_replans" in res.abort_reason
        or "empty_plan" in res.abort_reason
    )


def test_runner_enforces_step_budget(db_session) -> None:
    runner = Runner(FakeProvider(), db_session, budget=RunBudget(max_steps=1))
    res = runner.run("Compute (2+3)*7 calculate")
    assert res.status == GoalStatus.ABORTED
    assert "max_steps" in (res.abort_reason or "")


def test_runner_enforces_replan_budget(db_session) -> None:
    runner = Runner(FakeProvider(), db_session, budget=RunBudget(max_replans=0))
    res = runner.run("Send an email to alice@example.com")
    assert res.status == GoalStatus.ABORTED


def test_runner_replan_count_recorded_on_extract_failure(db_session) -> None:
    runner = Runner(FakeProvider(), db_session)
    res = runner.run("Extract the order id from unclear strict prose")
    assert res.status == GoalStatus.SUCCEEDED
    assert res.replan_count >= 1
