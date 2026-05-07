"""E2E test — runs the eval suite and asserts the committed baseline matches."""

from __future__ import annotations

from pathlib import Path

from agentic_runner.eval_harness import (
    assert_matches_baseline,
    run_suite,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_eval_runner_v1_matches_baseline() -> None:
    suite_path = REPO_ROOT / "eval" / "suites" / "runner_v1.yaml"
    baseline_path = REPO_ROOT / "eval" / "baselines" / "runner_v1_fake.json"
    report = run_suite(suite_path, provider_name="fake")
    assert_matches_baseline(report, baseline_path)


def test_eval_metrics_floor() -> None:
    """Sanity floor — at least one replan and one abort exercised."""
    suite_path = REPO_ROOT / "eval" / "suites" / "runner_v1.yaml"
    report = run_suite(suite_path, provider_name="fake")
    metrics = report["metrics"]
    assert metrics["n"] == 15
    assert metrics["success_rate"] >= 0.85
    assert metrics["matches_expected_rate"] >= 0.95
    assert metrics["replan_rate"] > 0.0
    assert metrics["abort_rate"] > 0.0
