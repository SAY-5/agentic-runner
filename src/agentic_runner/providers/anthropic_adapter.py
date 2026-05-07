"""Anthropic tool-use adapter (live-mode only)."""

from __future__ import annotations

from agentic_runner.providers.base import (
    ChatMessage,
    ChatResponse,
    ToolSpec,
)


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str | None = None, model: str = "claude-3-5-sonnet-latest") -> None:
        self._api_key = api_key
        self._model = model

    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
    ) -> ChatResponse:
        raise NotImplementedError(
            "Anthropic adapter requires the anthropic SDK and ANTHROPIC_API_KEY; "
            "use --provider fake in CI."
        )
