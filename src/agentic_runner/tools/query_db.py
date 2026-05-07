"""Read-only SQL tool against the bundled demo schema."""

from __future__ import annotations

import re
import sqlite3
import threading
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from agentic_runner.tools._base import ToolInvocationError, register_tool


class QueryDbInput(BaseModel):
    sql: str = Field(min_length=1, max_length=2000)


class QueryDbOutput(BaseModel):
    rows: list[dict[str, Any]]
    row_count: int


_SETUP_SQL = """
CREATE TABLE IF NOT EXISTS departments (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT NOT NULL,
    salary REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY,
    employee_id INTEGER NOT NULL,
    amount REAL NOT NULL,
    placed_on TEXT NOT NULL
);

INSERT OR IGNORE INTO departments (id, name) VALUES
    (1, 'engineering'),
    (2, 'sales'),
    (3, 'support');

INSERT OR IGNORE INTO employees (id, name, department, salary) VALUES
    (1, 'Ada Lovelace', 'engineering', 110000),
    (2, 'Grace Hopper', 'engineering', 105000),
    (3, 'Linus Torvalds', 'engineering', 90000),
    (4, 'Don Draper', 'sales', 70000),
    (5, 'Peggy Olson', 'sales', 80000),
    (6, 'Stan Marsh', 'support', 50000);

INSERT OR IGNORE INTO orders (id, employee_id, amount, placed_on) VALUES
    (101, 4, 300.00, '2024-01-15'),
    (102, 5, 450.00, '2024-02-02'),
    (103, 4, 150.00, '2024-02-20'),
    (104, 5, 225.00, '2024-03-01');
"""


_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        c = sqlite3.connect(":memory:", check_same_thread=False)
        c.row_factory = sqlite3.Row
        c.executescript(_SETUP_SQL)
        c.commit()
        _conn = c
    return _conn


def reset_demo_db() -> None:
    """Drop the cached connection (used in tests)."""
    global _conn
    if _conn is not None:
        _conn.close()
    _conn = None


_SELECT_RE = re.compile(r"^\s*select\b", re.IGNORECASE)
_DENY_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum)\b",
    re.IGNORECASE,
)


@register_tool
class QueryDbTool:
    name: ClassVar[str] = "query_db"
    description: ClassVar[str] = (
        "Run a read-only SELECT against the demo schema (employees, departments, orders)."
    )
    input_model: ClassVar[type[BaseModel]] = QueryDbInput
    output_model: ClassVar[type[BaseModel]] = QueryDbOutput
    max_runtime_ms: ClassVar[int] = 500
    idempotent: ClassVar[bool] = True

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        parsed = QueryDbInput.model_validate(args)
        sql = parsed.sql.strip().rstrip(";")
        if not _SELECT_RE.search(sql):
            raise ToolInvocationError("query_db: only SELECT statements are allowed")
        if _DENY_RE.search(sql):
            raise ToolInvocationError("query_db: write/DDL keywords are not permitted")
        if ";" in sql:
            raise ToolInvocationError("query_db: multi-statement queries are not allowed")

        with _lock:
            conn = _get_conn()
            try:
                cur = conn.execute(sql)
            except sqlite3.Error as exc:
                raise ToolInvocationError(f"query_db: {exc}") from exc
            rows = [dict(r) for r in cur.fetchall()]
        return {"rows": rows, "row_count": len(rows)}
