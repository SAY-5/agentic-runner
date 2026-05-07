"""Validator tests — schema mismatch and confidence threshold."""

from __future__ import annotations

from pydantic import BaseModel

from agentic_runner.failure import FailureKind
from agentic_runner.validator import Validator


class _ExpectedOutput(BaseModel):
    value: float
    confidence: float | None = None


def test_validator_passes_valid_output() -> None:
    out = Validator().validate("calc", {"value": 1.0}, _ExpectedOutput)
    assert out.valid is True
    assert out.failure is None


def test_validator_flags_schema_mismatch() -> None:
    out = Validator().validate("calc", {"value": "not_a_number"}, _ExpectedOutput)
    assert out.valid is False
    assert out.failure is not None
    assert out.failure.kind == FailureKind.OUTPUT_SCHEMA_MISMATCH
    assert out.failure.schema_violations


def test_validator_flags_low_confidence() -> None:
    out = Validator().validate(
        "calc", {"value": 1.0, "confidence": 0.3}, _ExpectedOutput, confidence_threshold=0.7
    )
    assert out.valid is False
    assert out.failure is not None
    assert out.failure.kind == FailureKind.CONFIDENCE_TOO_LOW


def test_validator_no_threshold_allows_any_confidence() -> None:
    out = Validator().validate("calc", {"value": 1.0, "confidence": 0.1}, _ExpectedOutput)
    assert out.valid is True
