"""Validator — checks tool output against its declared output schema."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from agentic_runner.failure import FailureKind, FailureReason


class ValidationOutcome(BaseModel):
    valid: bool
    failure: FailureReason | None = None


class Validator:
    """Stateless wrapper around Pydantic schema validation."""

    def validate(
        self,
        tool_name: str,
        output: dict[str, Any],
        output_model: type[BaseModel],
        confidence_threshold: float = 0.0,
    ) -> ValidationOutcome:
        try:
            output_model.model_validate(output)
        except ValidationError as exc:
            violations = [
                {
                    "loc": list(err.get("loc", [])),
                    "type": err.get("type"),
                    "msg": err.get("msg"),
                }
                for err in exc.errors()
            ]
            return ValidationOutcome(
                valid=False,
                failure=FailureReason(
                    kind=FailureKind.OUTPUT_SCHEMA_MISMATCH,
                    message=f"{tool_name} output failed schema",
                    tool_name=tool_name,
                    schema_violations=violations,
                ),
            )

        confidence = output.get("confidence")
        if (
            confidence_threshold > 0.0
            and isinstance(confidence, int | float)
            and float(confidence) < confidence_threshold
        ):
            return ValidationOutcome(
                valid=False,
                failure=FailureReason(
                    kind=FailureKind.CONFIDENCE_TOO_LOW,
                    message=(
                        f"{tool_name} confidence {confidence} below threshold "
                        f"{confidence_threshold}"
                    ),
                    tool_name=tool_name,
                    details={"confidence": float(confidence)},
                ),
            )

        return ValidationOutcome(valid=True)
