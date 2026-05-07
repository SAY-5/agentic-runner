"""CLI smoke tests via Click's testing helpers."""

from __future__ import annotations

import json
import os

import pytest
from click.testing import CliRunner

from agentic_runner.cli import cli

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="set RUN_INTEGRATION=1 to enable",
)


def test_cli_run_calculate(tmp_path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["run", "--goal", "Compute (2+3)*7 with calculate", "--provider", "fake"],
    )
    assert result.exit_code == 0, result.output
    # structlog may interleave JSON log lines on stdout; find the final JSON
    # payload (the one starting with "{\n").
    output = result.output
    start = output.find("{\n")
    assert start >= 0, output
    payload = json.loads(output[start:])
    assert payload["status"] == "succeeded"
    assert "35" in (payload["final_result"] or "")


def test_cli_eval_smoke(tmp_path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "eval",
            "smoke",
            "--suite",
            "runner_v1",
            "--baseline",
            "eval/baselines/runner_v1_fake.json",
            "--suite-dir",
            "eval/suites",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "baseline match" in result.output


def test_cli_seed(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "seed.sqlite"
    monkeypatch.setenv("AGENTIC_RUNNER_DATABASE_URL", f"sqlite:///{db_path}")
    runner = CliRunner()
    result = runner.invoke(cli, ["seed"])
    assert result.exit_code == 0
    assert db_path.exists()
