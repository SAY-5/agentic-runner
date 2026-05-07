"""initial schema

Revision ID: 20260507_0000
Revises:
Create Date: 2026-05-07 00:00:00

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260507_0000"
down_revision = None
branch_labels = None
depends_on = None


goal_status = sa.Enum(
    "pending",
    "running",
    "succeeded",
    "failed",
    "aborted",
    name="goal_status",
)
subtask_status = sa.Enum(
    "pending",
    "running",
    "done",
    "failed",
    "skipped",
    name="subtask_status",
)
tool_call_status = sa.Enum(
    "ok",
    "error",
    "timeout",
    name="tool_call_status",
)


def upgrade() -> None:
    op.create_table(
        "goals",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("goal_text", sa.Text, nullable=False),
        sa.Column("status", goal_status, nullable=False, server_default="pending"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("total_cost_usd", sa.Float, nullable=False, server_default="0"),
        sa.Column("total_steps", sa.Integer, nullable=False, server_default="0"),
        sa.Column("replan_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("abort_reason", sa.Text, nullable=True),
        sa.Column("final_result", sa.Text, nullable=True),
        sa.Column("max_steps", sa.Integer, nullable=False, server_default="20"),
        sa.Column("max_replans", sa.Integer, nullable=False, server_default="5"),
        sa.Column("max_cost_usd", sa.Float, nullable=False, server_default="0.5"),
    )
    op.create_table(
        "subtasks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "goal_id",
            sa.String(36),
            sa.ForeignKey("goals.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "parent_subtask_id",
            sa.String(36),
            sa.ForeignKey("subtasks.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("idx", sa.Integer, nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("status", subtask_status, nullable=False, server_default="pending"),
        sa.Column("confidence_threshold", sa.Float, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_subtasks_goal_idx", "subtasks", ["goal_id", "idx"])

    op.create_table(
        "tool_calls",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "subtask_id",
            sa.String(36),
            sa.ForeignKey("subtasks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("idx", sa.Integer, nullable=False),
        sa.Column("tool_name", sa.String(64), nullable=False),
        sa.Column("args", sa.JSON, nullable=True),
        sa.Column("output", sa.JSON, nullable=True),
        sa.Column("latency_ms", sa.Integer, nullable=False, server_default="0"),
        sa.Column("cost_usd", sa.Float, nullable=False, server_default="0"),
        sa.Column("status", tool_call_status, nullable=False, server_default="ok"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_tool_calls_subtask_idx", "tool_calls", ["subtask_id", "idx"])

    op.create_table(
        "validation_results",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "tool_call_id",
            sa.String(36),
            sa.ForeignKey("tool_calls.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("valid", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("failure_reason", sa.String(64), nullable=True),
        sa.Column("schema_violations", sa.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "replan_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "goal_id",
            sa.String(36),
            sa.ForeignKey("goals.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("idx", sa.Integer, nullable=False),
        sa.Column(
            "triggered_by_tool_call_id",
            sa.String(36),
            sa.ForeignKey("tool_calls.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("failure_reason", sa.String(64), nullable=False),
        sa.Column("new_plan_summary", sa.Text, nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_replan_events_goal_idx", "replan_events", ["goal_id", "idx"])


def downgrade() -> None:
    op.drop_index("ix_replan_events_goal_idx", table_name="replan_events")
    op.drop_table("replan_events")
    op.drop_table("validation_results")
    op.drop_index("ix_tool_calls_subtask_idx", table_name="tool_calls")
    op.drop_table("tool_calls")
    op.drop_index("ix_subtasks_goal_idx", table_name="subtasks")
    op.drop_table("subtasks")
    op.drop_table("goals")
    sa.Enum(name="tool_call_status").drop(op.get_bind(), checkfirst=False)
    sa.Enum(name="subtask_status").drop(op.get_bind(), checkfirst=False)
    sa.Enum(name="goal_status").drop(op.get_bind(), checkfirst=False)
