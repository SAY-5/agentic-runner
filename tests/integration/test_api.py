"""Integration test — exercise the FastAPI surface."""

from __future__ import annotations

import os
import time

import pytest
from fastapi.testclient import TestClient

from agentic_runner import models
from agentic_runner.api import create_app
from agentic_runner.models import init_engine

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="set RUN_INTEGRATION=1 to enable",
)


@pytest.fixture()
def client(tmp_path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "api_test.sqlite"
    url = f"sqlite:///{db_path}"
    monkeypatch.setenv("AGENTIC_RUNNER_DATABASE_URL", url)
    monkeypatch.setenv("AGENTIC_RUNNER_PROVIDER", "fake")
    init_engine(url)
    models.Base.metadata.create_all(models._engine)  # type: ignore[arg-type]
    app = create_app()
    return TestClient(app)


def test_healthz(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_create_goal_and_poll(client: TestClient) -> None:
    r = client.post(
        "/v1/goals",
        json={"goal_text": "Compute (2+3)*7 with calculate", "max_steps": 10},
    )
    assert r.status_code == 202
    body = r.json()
    gid = body["goal_id"]

    final = None
    for _ in range(50):
        time.sleep(0.1)
        rr = client.get(f"/v1/goals/{gid}")
        assert rr.status_code == 200
        if rr.json()["status"] in {"succeeded", "aborted", "failed"}:
            final = rr.json()
            break
    assert final is not None
    assert final["status"] == "succeeded"

    tr = client.get(f"/v1/goals/{gid}/trace")
    assert tr.status_code == 200
    assert tr.json()["status"] == "succeeded"


def test_goal_not_found(client: TestClient) -> None:
    r = client.get("/v1/goals/no-such-id")
    assert r.status_code == 404


def test_list_goals(client: TestClient) -> None:
    r = client.get("/v1/goals")
    assert r.status_code == 200
    assert "goals" in r.json()
