"""SQLAlchemy ORM models for goals, subtasks, tool calls, validations, replans."""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class GoalStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"


class SubtaskStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class ToolCallStatus(str, enum.Enum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"


class Goal(Base):
    __tablename__ = "goals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    goal_text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[GoalStatus] = mapped_column(
        Enum(GoalStatus, name="goal_status"), default=GoalStatus.PENDING, nullable=False
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    total_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    total_steps: Mapped[int] = mapped_column(Integer, default=0)
    replan_count: Mapped[int] = mapped_column(Integer, default=0)
    abort_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_result: Mapped[str | None] = mapped_column(Text, nullable=True)
    max_steps: Mapped[int] = mapped_column(Integer, default=20)
    max_replans: Mapped[int] = mapped_column(Integer, default=5)
    max_cost_usd: Mapped[float] = mapped_column(Float, default=0.50)

    subtasks: Mapped[list[Subtask]] = relationship(
        back_populates="goal", cascade="all, delete-orphan", order_by="Subtask.idx"
    )
    replan_events: Mapped[list[ReplanEvent]] = relationship(
        back_populates="goal", cascade="all, delete-orphan", order_by="ReplanEvent.idx"
    )


class Subtask(Base):
    __tablename__ = "subtasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    goal_id: Mapped[str] = mapped_column(ForeignKey("goals.id", ondelete="CASCADE"), nullable=False)
    parent_subtask_id: Mapped[str | None] = mapped_column(
        ForeignKey("subtasks.id", ondelete="SET NULL"), nullable=True
    )
    idx: Mapped[int] = mapped_column(Integer, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[SubtaskStatus] = mapped_column(
        Enum(SubtaskStatus, name="subtask_status"), default=SubtaskStatus.PENDING, nullable=False
    )
    confidence_threshold: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    goal: Mapped[Goal] = relationship(back_populates="subtasks")
    tool_calls: Mapped[list[ToolCall]] = relationship(
        back_populates="subtask", cascade="all, delete-orphan", order_by="ToolCall.idx"
    )

    __table_args__ = (Index("ix_subtasks_goal_idx", "goal_id", "idx"),)


class ToolCall(Base):
    __tablename__ = "tool_calls"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    subtask_id: Mapped[str] = mapped_column(
        ForeignKey("subtasks.id", ondelete="CASCADE"), nullable=False
    )
    idx: Mapped[int] = mapped_column(Integer, nullable=False)
    tool_name: Mapped[str] = mapped_column(String(64), nullable=False)
    args: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    output: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[ToolCallStatus] = mapped_column(
        Enum(ToolCallStatus, name="tool_call_status"), default=ToolCallStatus.OK, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    subtask: Mapped[Subtask] = relationship(back_populates="tool_calls")
    validation: Mapped[ValidationResult | None] = relationship(
        back_populates="tool_call", cascade="all, delete-orphan", uselist=False
    )

    __table_args__ = (Index("ix_tool_calls_subtask_idx", "subtask_id", "idx"),)


class ValidationResult(Base):
    __tablename__ = "validation_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    tool_call_id: Mapped[str] = mapped_column(
        ForeignKey("tool_calls.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    valid: Mapped[bool] = mapped_column(default=True, nullable=False)
    failure_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    schema_violations: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    tool_call: Mapped[ToolCall] = relationship(back_populates="validation")


class ReplanEvent(Base):
    __tablename__ = "replan_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    goal_id: Mapped[str] = mapped_column(ForeignKey("goals.id", ondelete="CASCADE"), nullable=False)
    idx: Mapped[int] = mapped_column(Integer, nullable=False)
    triggered_by_tool_call_id: Mapped[str | None] = mapped_column(
        ForeignKey("tool_calls.id", ondelete="SET NULL"), nullable=True
    )
    failure_reason: Mapped[str] = mapped_column(String(64), nullable=False)
    new_plan_summary: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    goal: Mapped[Goal] = relationship(back_populates="replan_events")

    __table_args__ = (Index("ix_replan_events_goal_idx", "goal_id", "idx"),)


_engine = None
_SessionLocal: sessionmaker[Session] | None = None


def init_engine(database_url: str) -> None:
    global _engine, _SessionLocal
    _engine = create_engine(database_url, pool_pre_ping=True, future=True)
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)


def get_session() -> Session:
    if _SessionLocal is None:
        from agentic_runner.settings import get_settings

        init_engine(get_settings().database_url)
    assert _SessionLocal is not None
    return _SessionLocal()


def create_all_tables() -> None:
    """Create tables — used for tests and SQLite fallback."""
    if _engine is None:
        from agentic_runner.settings import get_settings

        init_engine(get_settings().database_url)
    assert _engine is not None
    Base.metadata.create_all(_engine)
