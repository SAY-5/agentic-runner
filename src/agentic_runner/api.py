"""FastAPI surface."""

from __future__ import annotations

import threading
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import desc, select
from sqlalchemy.exc import SQLAlchemyError

from agentic_runner import models
from agentic_runner.models import Goal, GoalStatus, get_session, init_engine
from agentic_runner.providers import build_provider
from agentic_runner.runner import RunBudget, Runner
from agentic_runner.runner import _trace_for_goal as build_trace
from agentic_runner.settings import get_settings
from agentic_runner.trace import configure_logging


class CreateGoalRequest(BaseModel):
    goal_text: str = Field(min_length=1, max_length=4_000)
    max_steps: int | None = Field(default=None, ge=1, le=100)
    max_cost_usd: float | None = Field(default=None, ge=0.0, le=10.0)
    max_replans: int | None = Field(default=None, ge=0, le=20)


class CreateGoalResponse(BaseModel):
    goal_id: str
    status_url: str


class GoalSummary(BaseModel):
    id: str
    status: str
    goal_text: str
    final_result: str | None
    abort_reason: str | None
    total_steps: int
    replan_count: int
    total_cost_usd: float


class GoalListResponse(BaseModel):
    goals: list[GoalSummary]
    next_cursor: str | None


class TraceResponse(BaseModel):
    goal_id: str
    status: str
    trace: list[dict[str, Any]]


def create_app() -> FastAPI:
    configure_logging(get_settings().log_level)
    init_engine(get_settings().database_url)

    app = FastAPI(title="agentic-runner", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        try:
            with get_session() as s:
                s.execute(select(1))
            return {"status": "ok"}
        except SQLAlchemyError as exc:
            raise HTTPException(status_code=503, detail=f"db_unhealthy: {exc}") from exc

    @app.post("/v1/goals", response_model=CreateGoalResponse, status_code=202)
    def create_goal(
        req: CreateGoalRequest, background_tasks: BackgroundTasks
    ) -> CreateGoalResponse:
        with get_session() as s:
            goal = Goal(
                goal_text=req.goal_text,
                max_steps=req.max_steps or 20,
                max_replans=req.max_replans if req.max_replans is not None else 5,
                max_cost_usd=req.max_cost_usd if req.max_cost_usd is not None else 0.50,
            )
            s.add(goal)
            s.commit()
            s.refresh(goal)
            gid = goal.id

        def _run() -> None:
            settings = get_settings()
            provider = build_provider(settings.provider)
            from agentic_runner.models import _SessionLocal as factory  # noqa: PLC0415

            assert factory is not None
            budget = RunBudget(
                max_steps=req.max_steps or 20,
                max_replans=req.max_replans if req.max_replans is not None else 5,
                max_cost_usd=req.max_cost_usd if req.max_cost_usd is not None else 0.50,
            )
            runner = Runner(provider, factory, budget=budget)
            try:
                runner.run(req.goal_text, goal_id=gid)
            except Exception:  # noqa: BLE001
                with get_session() as s2:
                    g = s2.get(Goal, gid)
                    if g is not None:
                        g.status = GoalStatus.FAILED
                        s2.commit()

        threading.Thread(target=_run, daemon=True).start()
        return CreateGoalResponse(goal_id=gid, status_url=f"/v1/goals/{gid}")

    @app.get("/v1/goals", response_model=GoalListResponse)
    def list_goals(
        status: str | None = Query(default=None),
        cursor: str | None = Query(default=None),
        limit: int = Query(default=20, ge=1, le=100),
    ) -> GoalListResponse:
        with get_session() as s:
            stmt = select(Goal).order_by(desc(Goal.started_at))
            if status:
                try:
                    stmt = stmt.where(Goal.status == GoalStatus(status))
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
            if cursor:
                anchor = s.get(Goal, cursor)
                if anchor is not None:
                    stmt = stmt.where(Goal.started_at < anchor.started_at)
            rows = s.scalars(stmt.limit(limit + 1)).all()
            next_cursor = None
            if len(rows) > limit:
                next_cursor = rows[limit - 1].id
                rows = rows[:limit]
            return GoalListResponse(
                goals=[_summary(g) for g in rows],
                next_cursor=next_cursor,
            )

    @app.get("/v1/goals/{goal_id}", response_model=GoalSummary)
    def get_goal(goal_id: str) -> GoalSummary:
        with get_session() as s:
            g = s.get(Goal, goal_id)
            if g is None:
                raise HTTPException(status_code=404, detail="goal not found")
            return _summary(g)

    @app.get("/v1/goals/{goal_id}/trace", response_model=TraceResponse)
    def get_trace(goal_id: str) -> TraceResponse:
        with get_session() as s:
            g = s.get(Goal, goal_id)
            if g is None:
                raise HTTPException(status_code=404, detail="goal not found")
            trace = build_trace(s, goal_id)
            return TraceResponse(goal_id=goal_id, status=g.status.value, trace=trace)

    return app


def _summary(g: Goal) -> GoalSummary:
    return GoalSummary(
        id=g.id,
        status=g.status.value,
        goal_text=g.goal_text,
        final_result=g.final_result,
        abort_reason=g.abort_reason,
        total_steps=g.total_steps,
        replan_count=g.replan_count,
        total_cost_usd=g.total_cost_usd,
    )


app = create_app()


__all__ = ["app", "create_app"]


# Keep models import alive.
_ = models
