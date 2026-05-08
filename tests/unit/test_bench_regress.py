"""Unit tests for the bench-regress drift gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_runner.eval_harness import assert_within_drift


def _write_baseline(tmp_path: Path, metrics: dict[str, float]) -> Path:
    p = tmp_path / "baseline.json"
    p.write_text(json.dumps({"metrics": metrics, "results": []}))
    return p


def test_within_drift_accepts_identical_metrics(tmp_path: Path) -> None:
    metrics = {
        "success_rate": 0.95,
        "abort_rate": 0.05,
        "avg_steps": 4.1,
        "avg_cost_usd": 0.00268,
        "rubric_pass_rate": 1.0,
        "tool_sequence_jaccard_avg": 1.0,
    }
    baseline = _write_baseline(tmp_path, metrics)
    lines = assert_within_drift({"metrics": metrics}, baseline, max_drift=0.30)
    assert any("OK" in line for line in lines)


def test_within_drift_accepts_small_movement(tmp_path: Path) -> None:
    base = {"success_rate": 0.90, "avg_steps": 4.0, "avg_cost_usd": 0.003}
    baseline = _write_baseline(tmp_path, base)
    new = {"metrics": {"success_rate": 0.95, "avg_steps": 4.4, "avg_cost_usd": 0.0035}}
    # Drifts here are roughly 5.6%, 10%, 16.7% — all under 30%.
    assert_within_drift(new, baseline, max_drift=0.30)


def test_within_drift_trips_on_large_movement(tmp_path: Path) -> None:
    base = {"success_rate": 1.0, "avg_steps": 4.0}
    baseline = _write_baseline(tmp_path, base)
    new = {"metrics": {"success_rate": 0.5, "avg_steps": 4.0}}
    with pytest.raises(AssertionError, match="success_rate"):
        assert_within_drift(new, baseline, max_drift=0.30)


def test_within_drift_handles_zero_baseline(tmp_path: Path) -> None:
    base = {"abort_rate": 0.0, "avg_steps": 4.0}
    baseline = _write_baseline(tmp_path, base)
    # Going from 0 to anything trips the gate (drift -> infinity).
    new = {"metrics": {"abort_rate": 0.10, "avg_steps": 4.0}}
    with pytest.raises(AssertionError):
        assert_within_drift(new, baseline, max_drift=0.30)
