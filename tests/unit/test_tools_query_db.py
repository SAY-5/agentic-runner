"""Tests for the read-only query_db tool."""

from __future__ import annotations

import pytest

from agentic_runner.tools._base import ToolInvocationError
from agentic_runner.tools.query_db import QueryDbTool


def test_query_db_select_employees() -> None:
    out = QueryDbTool().invoke({"sql": "SELECT name FROM employees"})
    assert out["row_count"] == 6
    assert {row["name"] for row in out["rows"]} >= {"Ada Lovelace", "Grace Hopper"}


def test_query_db_avg_engineering_salary() -> None:
    out = QueryDbTool().invoke(
        {"sql": "SELECT AVG(salary) AS avg_salary FROM employees WHERE department='engineering'"}
    )
    assert out["row_count"] == 1
    assert abs(out["rows"][0]["avg_salary"] - (110000 + 105000 + 90000) / 3) < 1e-6


def test_query_db_rejects_insert() -> None:
    with pytest.raises(ToolInvocationError):
        QueryDbTool().invoke({"sql": "INSERT INTO employees (id) VALUES (99)"})


def test_query_db_rejects_drop() -> None:
    with pytest.raises(ToolInvocationError):
        QueryDbTool().invoke({"sql": "DROP TABLE employees"})


def test_query_db_rejects_multi_statement() -> None:
    with pytest.raises(ToolInvocationError):
        QueryDbTool().invoke({"sql": "SELECT 1; SELECT 2"})


def test_query_db_rejects_attach() -> None:
    with pytest.raises(ToolInvocationError):
        QueryDbTool().invoke({"sql": "ATTACH DATABASE 'evil.db' AS evil"})


def test_query_db_rejects_create() -> None:
    with pytest.raises(ToolInvocationError):
        QueryDbTool().invoke({"sql": "CREATE TABLE leak (id INT)"})
