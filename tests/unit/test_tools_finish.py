"""Tests for the terminal finish tool."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentic_runner.tools.finish import FinishTool


def test_finish_returns_final() -> None:
    assert FinishTool().invoke({"result": "done"})["final"] == "done"


def test_finish_rejects_empty() -> None:
    with pytest.raises(ValidationError):
        FinishTool().invoke({"result": ""})
