"""Integration test — full goal end-to-end against FakeProvider, twice."""

from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from agentic_runner import models
from agentic_runner.providers.fake import FakeProvider
from agentic_runner.runner import RunBudget, Runner

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="set RUN_INTEGRATION=1 to enable",
)


def _strip_ids(trace: list[dict]) -> list[dict]:
    out = []
    for top in trace:
        sts = []
        for st in top["subtasks"]:
            tcs = []
            for tc in st["tool_calls"]:
                tcs.append(
                    {
                        "tool": tc["tool"],
                        "args": tc["args"],
                        "output": tc["output"],
                        "status": tc["status"],
                    }
                )
            sts.append(
                {"description": st["description"], "status": st["status"], "tool_calls": tcs}
            )
        out.append({"subtasks": sts})
    return out


def _factory():
    engine = create_engine("sqlite://", future=True)
    models.Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def test_replay_determinism() -> None:
    factory_a = _factory()
    factory_b = _factory()
    goal = "Compute (2+3)*7 with calculate"
    a = Runner(FakeProvider(), factory_a, RunBudget()).run(goal)
    b = Runner(FakeProvider(), factory_b, RunBudget()).run(goal)
    assert _strip_ids(a.trace) == _strip_ids(b.trace)
    assert a.replan_count == b.replan_count
    assert a.total_steps == b.total_steps
    assert abs(a.total_cost_usd - b.total_cost_usd) < 1e-9


def test_replay_replan_path() -> None:
    factory_a = _factory()
    factory_b = _factory()
    goal = "Extract the order id from unclear strict prose"
    a = Runner(FakeProvider(), factory_a, RunBudget()).run(goal)
    b = Runner(FakeProvider(), factory_b, RunBudget()).run(goal)
    assert _strip_ids(a.trace) == _strip_ids(b.trace)
    assert a.replan_count == b.replan_count == 1
