"""Click-based CLI: run a single goal, run/smoke evals, seed the demo DB."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from agentic_runner.eval_harness import (
    assert_matches_baseline,
    assert_within_drift,
    run_suite,
    write_baseline,
)
from agentic_runner.models import create_all_tables, init_engine
from agentic_runner.providers import build_provider
from agentic_runner.runner import RunBudget, Runner
from agentic_runner.settings import get_settings
from agentic_runner.trace import configure_logging


@click.group()
def cli() -> None:
    """agentic-runner — goal decomposition with plan-validate-replan loop."""
    configure_logging(get_settings().log_level)


@cli.command()
@click.option("--goal", "goal_text", required=True, help="Goal text to run")
@click.option("--provider", default=None, help="Provider name (fake|openai|anthropic)")
@click.option("--max-steps", default=20, type=int)
@click.option("--max-replans", default=5, type=int)
@click.option("--max-cost-usd", default=0.50, type=float)
@click.option("--db-url", default=None, help="Override database URL (default: in-memory sqlite)")
def run(
    goal_text: str,
    provider: str | None,
    max_steps: int,
    max_replans: int,
    max_cost_usd: float,
    db_url: str | None,
) -> None:
    """Run one goal and print the result + trace as JSON."""
    settings = get_settings()
    provider_name = provider or settings.provider
    if db_url is None:
        db_url = "sqlite:///:memory:"
    init_engine(db_url)
    create_all_tables()

    from agentic_runner.models import _SessionLocal as factory  # noqa: PLC0415

    assert factory is not None
    p = build_provider(provider_name)
    budget = RunBudget(max_steps=max_steps, max_replans=max_replans, max_cost_usd=max_cost_usd)
    runner = Runner(p, factory, budget=budget)
    result = runner.run(goal_text)
    payload = {
        "status": result.status.value,
        "final_result": result.final_result,
        "abort_reason": result.abort_reason,
        "total_steps": result.total_steps,
        "replan_count": result.replan_count,
        "total_cost_usd": round(result.total_cost_usd, 6),
        "trace": result.trace,
    }
    click.echo(json.dumps(payload, indent=2))


@cli.group()
def eval() -> None:
    """Eval commands."""


@eval.command("run")
@click.option("--suite", default="runner_v1", help="Suite name (matches eval/suites/<name>.yaml)")
@click.option("--provider", default="fake")
@click.option(
    "--output",
    "output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Where to write the baseline JSON",
)
@click.option(
    "--suite-dir",
    type=click.Path(file_okay=False, exists=True, path_type=Path),
    default=Path("eval/suites"),
)
def eval_run(suite: str, provider: str, output: Path | None, suite_dir: Path) -> None:
    """Run an eval suite and (optionally) write the baseline JSON."""
    suite_path = suite_dir / f"{suite}.yaml"
    report = run_suite(suite_path, provider_name=provider)
    if output is not None:
        write_baseline(report, output)
        click.echo(f"wrote baseline: {output}")
    metrics = report["metrics"]
    click.echo(json.dumps(metrics, indent=2))


@eval.command("smoke")
@click.option("--suite", default="runner_v1")
@click.option("--provider", default="fake")
@click.option(
    "--baseline",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--suite-dir",
    type=click.Path(file_okay=False, exists=True, path_type=Path),
    default=Path("eval/suites"),
)
def eval_smoke(suite: str, provider: str, baseline: Path, suite_dir: Path) -> None:
    """Re-run the suite and assert it matches the committed baseline."""
    suite_path = suite_dir / f"{suite}.yaml"
    report = run_suite(suite_path, provider_name=provider)
    try:
        assert_matches_baseline(report, baseline)
    except AssertionError as exc:
        click.echo(str(exc), err=True)
        sys.exit(2)
    click.echo("eval-smoke: baseline match ok")


@eval.command("bench-regress")
@click.option("--suite", default="runner_v1")
@click.option("--provider", default="fake")
@click.option(
    "--baseline",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--suite-dir",
    type=click.Path(file_okay=False, exists=True, path_type=Path),
    default=Path("eval/suites"),
)
@click.option(
    "--max-drift",
    type=float,
    default=0.30,
    help="Maximum allowed relative drift for any aggregate metric",
)
def eval_bench_regress(
    suite: str, provider: str, baseline: Path, suite_dir: Path, max_drift: float
) -> None:
    """Re-run the suite and assert no aggregate metric drifted past ``--max-drift``."""
    suite_path = suite_dir / f"{suite}.yaml"
    report = run_suite(suite_path, provider_name=provider)
    try:
        lines = assert_within_drift(report, baseline, max_drift=max_drift)
    except AssertionError as exc:
        click.echo(str(exc), err=True)
        sys.exit(2)
    click.echo(f"bench-regress: all metrics within {max_drift:.0%} drift")
    for line in lines:
        click.echo(line)


@cli.command()
def seed() -> None:
    """Create the database tables (no-op if they already exist)."""
    init_engine(get_settings().database_url)
    create_all_tables()
    click.echo("ok")


if __name__ == "__main__":
    cli()
