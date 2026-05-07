"""Tests for summarize and extract_json — both backed by FakeProvider."""

from __future__ import annotations

import pytest

from agentic_runner.tools._base import ToolInvocationError
from agentic_runner.tools.extract_json import ExtractJsonTool
from agentic_runner.tools.summarize import SummarizeTool


def test_summarize_returns_word_count() -> None:
    out = SummarizeTool().invoke({"text": "hello world", "max_words": 20})
    assert out["word_count"] >= 1
    assert isinstance(out["summary"], str)


def test_summarize_strict_returns_short() -> None:
    out = SummarizeTool().invoke({"text": "hello world", "max_words": 5, "strict": True})
    assert out["word_count"] <= 5


def test_extract_json_strict_passes_validation() -> None:
    out = ExtractJsonTool().invoke(
        {
            "text": "any prose",
            "schema": {
                "type": "object",
                "properties": {"order_id": {"type": "integer"}},
                "required": ["order_id"],
            },
            "hint": "use the structured prompt",
        }
    )
    assert out["extracted"]["order_id"] == 7


def test_extract_json_non_strict_fails_validation() -> None:
    with pytest.raises(ToolInvocationError):
        ExtractJsonTool().invoke(
            {
                "text": "any prose",
                "schema": {
                    "type": "object",
                    "properties": {"order_id": {"type": "integer"}},
                    "required": ["order_id"],
                },
            }
        )
