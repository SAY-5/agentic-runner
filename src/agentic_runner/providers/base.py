"""ChatProvider Protocol + shared message/tool types.

The shape mirrors function-calling APIs (OpenAI's ``tool_calls`` and
Anthropic's ``tool_use`` content blocks) — ``tool_calls`` is the
load-bearing field, not free-text completion.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """One turn in the chat history."""

    role: str  # system | user | assistant | tool
    content: str = ""
    name: str | None = None  # tool name when role == "tool"
    tool_call_id: str | None = None  # references a previous tool_call when role == "tool"


class ToolSpec(BaseModel):
    """Tool description shipped to the LLM in function-calling format."""

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)


class ToolCallRequest(BaseModel):
    """A function call the model wants to make."""

    id: str
    name: str
    arguments: dict[str, Any]


class ChatResponse(BaseModel):
    """Provider response."""

    text: str = ""
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    model_version: str = ""

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class ChatProvider(Protocol):
    """Protocol every concrete provider implements."""

    name: str

    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
    ) -> ChatResponse: ...
