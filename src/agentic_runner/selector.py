"""Selector — picks a tool + argument set for a single subtask."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from agentic_runner.failure import FailureKind, FailureReason
from agentic_runner.providers.base import ChatMessage, ChatProvider
from agentic_runner.tools._base import REGISTRY


class ToolCallPlan(BaseModel):
    tool_name: str
    arguments: dict[str, Any]


class SelectorResult(BaseModel):
    tool_call: ToolCallPlan | None = None
    failure: FailureReason | None = None
    cost_usd: float = 0.0


class Selector:
    """Asks the model which tool to use, given the subtask + tool list."""

    def __init__(self, provider: ChatProvider) -> None:
        self._provider = provider

    def choose_tool(
        self,
        goal: str,
        subtask_description: str,
        tool_hint: str,
        args_override: dict[str, Any] | None = None,
    ) -> SelectorResult:
        if tool_hint not in REGISTRY:
            return SelectorResult(
                failure=FailureReason(
                    kind=FailureKind.NO_TOOL_FOR_SUBTASK,
                    message=f"requested tool {tool_hint!r} is not registered",
                    tool_name=tool_hint,
                )
            )

        prompt = f"SELECT: {tool_hint}\nSUBTASK: {subtask_description}"
        resp = self._provider.chat(
            [
                ChatMessage(role="user", content=goal),
                ChatMessage(role="user", content=prompt),
            ]
        )
        if not resp.tool_calls:
            return SelectorResult(
                failure=FailureReason(
                    kind=FailureKind.NO_TOOL_FOR_SUBTASK,
                    message="provider returned no tool_calls",
                    tool_name=tool_hint,
                ),
                cost_usd=resp.cost_usd,
            )
        chosen = resp.tool_calls[0]
        if chosen.name not in REGISTRY:
            return SelectorResult(
                failure=FailureReason(
                    kind=FailureKind.NO_TOOL_FOR_SUBTASK,
                    message=f"selected tool {chosen.name!r} not registered",
                    tool_name=chosen.name,
                ),
                cost_usd=resp.cost_usd,
            )

        merged_args = dict(chosen.arguments)
        if args_override:
            merged_args.update(args_override)

        return SelectorResult(
            tool_call=ToolCallPlan(tool_name=chosen.name, arguments=merged_args),
            cost_usd=resp.cost_usd,
        )
