"""subtask dependencies + parallel flag

Revision ID: 20260508_0001
Revises: 20260507_0000
Create Date: 2026-05-08 00:00:01

Adds two columns to the ``subtasks`` table that drive parallel
subtask execution:

* ``dependencies`` (JSON): list of subtask IDs that must complete before
  this one starts. Empty list means no dependencies.
* ``parallel`` (Boolean): when true (and dependencies are satisfied) the
  runner may launch this subtask concurrently with its siblings.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260508_0001"
down_revision = "20260507_0000"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "subtasks",
        sa.Column("dependencies", sa.JSON(), nullable=False, server_default="[]"),
    )
    op.add_column(
        "subtasks",
        sa.Column(
            "parallel", sa.Boolean(), nullable=False, server_default=sa.text("false")
        ),
    )


def downgrade() -> None:
    op.drop_column("subtasks", "parallel")
    op.drop_column("subtasks", "dependencies")
