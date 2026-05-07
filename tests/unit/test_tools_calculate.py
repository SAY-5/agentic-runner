"""Tests for the calculate tool — happy path and rejection of unsafe input."""

from __future__ import annotations

import pytest

from agentic_runner.tools._base import ToolInvocationError
from agentic_runner.tools.calculate import CalculateTool


def test_calculate_basic() -> None:
    out = CalculateTool().invoke({"expression": "(2+3)*7"})
    assert out["value"] == 35.0


def test_calculate_unary_and_pow() -> None:
    assert CalculateTool().invoke({"expression": "-2**3"})["value"] == -8.0
    assert CalculateTool().invoke({"expression": "+5"})["value"] == 5.0


def test_calculate_rejects_names() -> None:
    with pytest.raises(ToolInvocationError):
        CalculateTool().invoke({"expression": "x+1"})


def test_calculate_rejects_calls() -> None:
    with pytest.raises(ToolInvocationError):
        CalculateTool().invoke({"expression": "abs(-1)"})


def test_calculate_rejects_attr() -> None:
    with pytest.raises(ToolInvocationError):
        CalculateTool().invoke({"expression": "(1).real"})


def test_calculate_rejects_syntax_error() -> None:
    with pytest.raises(ToolInvocationError):
        CalculateTool().invoke({"expression": "1 +"})


def test_calculate_rejects_string_literal() -> None:
    with pytest.raises(ToolInvocationError):
        CalculateTool().invoke({"expression": "'evil'"})
