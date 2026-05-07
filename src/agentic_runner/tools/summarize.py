"""LLM-backed summarization tool."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, Field

from agentic_runner.providers import build_provider
from agentic_runner.providers.base import ChatMessage
from agentic_runner.settings import get_settings
from agentic_runner.tools._base import ToolInvocationError, register_tool


class SummarizeInput(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)
    max_words: int = Field(default=20, ge=1, le=200)
    strict: bool = Field(default=False)


class SummarizeOutput(BaseModel):
    summary: str
    word_count: int


@register_tool
class SummarizeTool:
    name: ClassVar[str] = "summarize"
    description: ClassVar[str] = "Summarize the provided text within max_words."
    input_model: ClassVar[type[BaseModel]] = SummarizeInput
    output_model: ClassVar[type[BaseModel]] = SummarizeOutput
    max_runtime_ms: ClassVar[int] = 3000
    idempotent: ClassVar[bool] = True

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        parsed = SummarizeInput.model_validate(args)
        provider = build_provider(get_settings().provider)
        prompt = (
            f"SUMMARIZE: {parsed.text}\n"
            f"MAX_WORDS={parsed.max_words}\n"
            f"STRICT={'1' if parsed.strict else '0'}"
        )
        resp = provider.chat([ChatMessage(role="user", content=prompt)])
        summary = resp.text.strip()
        if not summary:
            raise ToolInvocationError("summarize: provider returned empty text")
        word_count = len(summary.split())
        return {"summary": summary, "word_count": word_count}
