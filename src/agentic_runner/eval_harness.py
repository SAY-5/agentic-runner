"""Eval harness — loads a YAML suite, runs each goal, computes metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from agentic_runner import models
from agentic_runner.models import GoalStatus
from agentic_runner.providers import build_provider
from agentic_runner.runner import RunBudget, Runner
from agentic_runner.tools.read_file import _resolve_safe
from agentic_runner.tools.write_file import WriteFileTool


def _ensure_workspace_fixtures() -> None:
    """Create the small files referenced by some eval goals."""
    fixtures = {
        "notes.txt": "Reminder: review pull requests on Friday.",
        "report.txt": "Quarterly report content.",
        "long_doc.txt": (
            "The roadmap covers infrastructure modernization, observability "
            "investments, and a graduated rollout of the new agent runtime "
            "across the platform."
        ),
        "salaries.txt": "Ada Lovelace: 110000\nGrace Hopper: 105000\nJohn Doe: 88000\n",
    }
    for name, content in fixtures.items():
        target = _resolve_safe(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text(content, encoding="utf-8")
    _ = WriteFileTool


def load_suite(suite_path: Path) -> dict[str, Any]:
    with suite_path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "goals" not in data:
        raise ValueError(f"invalid suite: {suite_path}")
    return data


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def _tool_sequence(trace: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for top in trace:
        for st in top.get("subtasks", []):
            for tc in st.get("tool_calls", []):
                if tc.get("status") == "ok":
                    out.append(tc["tool"])
    return out


def _rubric_pass(goal_id: str, status: str, final: str | None, replan_count: int) -> bool:
    if status == "aborted":
        return goal_id == "g13_send_email_abort"
    final = final or ""
    expected_substrings = {
        "g01_calc_avg_eng": "average",
        "g02_calc_simple": "35",
        "g03_list_employees": "employees",
        "g04_count_orders": "order",
        "g05_read_notes": "Notes",
        "g06_write_then_read": "Report",
        "g07_fetch_status": "Homepage",
        "g08_summarize_doc": "summary",
        "g09_extract_struct": "Structured",
        "g10_math_two_step": "75",
        "g11_distinct_departments": "department",
        "g12_find_john_salary": "salaries",
        "g14_summarize_short": "summary",
        "g15_extract_strict": "Order id",
    }
    needle = expected_substrings.get(goal_id, "")
    return not (needle and needle.lower() not in final.lower())


def run_suite(suite_path: Path, provider_name: str = "fake") -> dict[str, Any]:
    suite = load_suite(suite_path)
    _ensure_workspace_fixtures()

    engine = create_engine("sqlite://", future=True)
    models.Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    results: list[dict[str, Any]] = []
    success_count = 0
    abort_count = 0
    steps_total = 0
    cost_total = 0.0
    rubric_pass_count = 0
    jaccard_total = 0.0

    for goal in suite["goals"]:
        p = build_provider(provider_name)
        budget = RunBudget(max_steps=20, max_replans=5, max_cost_usd=0.50)
        runner = Runner(p, factory, budget=budget)
        result = runner.run(goal["goal_text"])

        observed_seq = _tool_sequence(result.trace)
        expected_seq = goal.get("expected_tool_sequence") or []
        jaccard = _jaccard(observed_seq, expected_seq)
        jaccard_total += jaccard

        rubric_ok = _rubric_pass(
            goal["id"], result.status.value, result.final_result, result.replan_count
        )
        if rubric_ok:
            rubric_pass_count += 1

        if result.status == GoalStatus.SUCCEEDED:
            success_count += 1
        if result.status == GoalStatus.ABORTED:
            abort_count += 1
        steps_total += result.total_steps
        cost_total += result.total_cost_usd

        results.append(
            {
                "id": goal["id"],
                "goal_text": goal["goal_text"],
                "status": result.status.value,
                "final_result": result.final_result,
                "abort_reason": result.abort_reason,
                "total_steps": result.total_steps,
                "replan_count": result.replan_count,
                "total_cost_usd": round(result.total_cost_usd, 6),
                "tool_sequence": observed_seq,
                "expected_tool_sequence": expected_seq,
                "tool_sequence_jaccard": round(jaccard, 6),
                "rubric_pass": rubric_ok,
                "expected_status": goal.get("expected_status"),
                "expected_replan_count_min": goal.get("expected_replan_count_min", 0),
                "matches_expected": (
                    result.status.value == goal.get("expected_status")
                    and result.replan_count >= goal.get("expected_replan_count_min", 0)
                ),
            }
        )
        del p

    n = len(results)
    metrics = {
        "n": n,
        "success_rate": round(success_count / n, 6) if n else 0.0,
        "abort_rate": round(abort_count / n, 6) if n else 0.0,
        "replan_rate": (
            round(sum(1 for r in results if r["replan_count"] > 0) / n, 6) if n else 0.0
        ),
        "avg_steps": round(steps_total / n, 6) if n else 0.0,
        "avg_cost_usd": round(cost_total / n, 6) if n else 0.0,
        "tool_sequence_jaccard_avg": round(jaccard_total / n, 6) if n else 0.0,
        "rubric_pass_rate": round(rubric_pass_count / n, 6) if n else 0.0,
        "matches_expected_rate": (
            round(sum(1 for r in results if r["matches_expected"]) / n, 6) if n else 0.0
        ),
    }
    return {
        "suite": suite["name"],
        "version": suite.get("version", "?"),
        "provider": provider_name,
        "metrics": metrics,
        "results": results,
    }


def write_baseline(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")


def assert_matches_baseline(report: dict[str, Any], baseline_path: Path) -> None:
    """Compare a fresh report to the committed baseline. Float fields use 1e-6."""
    with baseline_path.open() as f:
        baseline = json.load(f)
    diffs = _diff(report, baseline, "$")
    if diffs:
        raise AssertionError("eval baseline mismatch:\n  " + "\n  ".join(diffs[:30]))


def _diff(a: Any, b: Any, path: str) -> list[str]:
    if type(a) is not type(b):
        return [f"{path}: type {type(a).__name__} != {type(b).__name__}"]
    if isinstance(a, dict):
        diffs: list[str] = []
        if a.keys() != b.keys():
            missing = b.keys() - a.keys()
            extra = a.keys() - b.keys()
            if missing:
                diffs.append(f"{path}: missing keys {sorted(missing)}")
            if extra:
                diffs.append(f"{path}: extra keys {sorted(extra)}")
        for k in a.keys() & b.keys():
            diffs.extend(_diff(a[k], b[k], f"{path}.{k}"))
        return diffs
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        diffs = []
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            diffs.extend(_diff(x, y, f"{path}[{i}]"))
        return diffs
    if isinstance(a, float):
        if not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6):
            return [f"{path}: {a} != {b}"]
        return []
    if a != b:
        return [f"{path}: {a!r} != {b!r}"]
    return []
