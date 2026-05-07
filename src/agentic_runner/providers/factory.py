"""Build provider instances by name."""

from __future__ import annotations

from agentic_runner.providers.base import ChatProvider


def build_provider(name: str) -> ChatProvider:
    """Return a configured :class:`ChatProvider` for the given name."""
    if name == "fake":
        from agentic_runner.providers.fake import FakeProvider

        return FakeProvider()
    if name == "openai":
        from agentic_runner.providers.openai_adapter import OpenAIProvider

        return OpenAIProvider()
    if name == "anthropic":
        from agentic_runner.providers.anthropic_adapter import AnthropicProvider

        return AnthropicProvider()
    raise ValueError(f"unknown provider: {name!r}")
