"""Terminal tool — signals goal completion."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, Field

from agentic_runner.tools._base import register_tool


class FinishInput(BaseModel):
    result: str = Field(min_length=1, max_length=8_000)


class FinishOutput(BaseModel):
    final: str


@register_tool
class FinishTool:
    name: ClassVar[str] = "finish"
    description: ClassVar[str] = (
        "Signal that the goal is complete and return the final result string."
    )
    input_model: ClassVar[type[BaseModel]] = FinishInput
    output_model: ClassVar[type[BaseModel]] = FinishOutput
    max_runtime_ms: ClassVar[int] = 50
    idempotent: ClassVar[bool] = True

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        parsed = FinishInput.model_validate(args)
        return {"final": parsed.result}
