"""OpenAI function-calling adapter (live-mode only).

The adapter is intentionally thin — full SDK integration is out of
scope here. The CI hermetic path always uses :class:`FakeProvider`.
"""

from __future__ import annotations

from agentic_runner.providers.base import (
    ChatMessage,
    ChatResponse,
    ToolSpec,
)


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini") -> None:
        self._api_key = api_key
        self._model = model

    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
    ) -> ChatResponse:
        raise NotImplementedError(
            "OpenAI adapter requires the openai SDK and OPENAI_API_KEY; "
            "use --provider fake in CI."
        )
