"""Tool Protocol + registry."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from agentic_runner.trace import get_logger, span

_log = get_logger("tools")


class ToolInvocationError(RuntimeError):
    """Raised when a tool's :meth:`invoke` fails functionally."""


class ToolTimeoutError(RuntimeError):
    """Raised when a tool exceeds its declared max_runtime_ms."""


@runtime_checkable
class Tool(Protocol):
    """Every tool exposes this surface."""

    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    max_runtime_ms: int
    idempotent: bool

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]: ...


REGISTRY: dict[str, Tool] = {}


def register_tool(cls: type) -> type:
    """Class decorator that instantiates the tool and adds it to the registry."""
    inst = cls()
    if not isinstance(inst, Tool):
        raise TypeError(f"{cls!r} does not satisfy the Tool protocol")
    if inst.name in REGISTRY:
        raise ValueError(f"tool already registered: {inst.name}")
    REGISTRY[inst.name] = inst
    return cls


def invoke_with_guard(tool: Tool, args: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Run a tool with input validation + soft runtime guard."""
    tool.input_model.model_validate(args)

    started = time.perf_counter()
    with span("tool.invoke", tool=tool.name):
        try:
            output = tool.invoke(args)
        except (ToolInvocationError, ToolTimeoutError):
            raise
        except Exception as exc:  # noqa: BLE001
            _log.warning("tool_failed", tool=tool.name, error=str(exc))
            raise ToolInvocationError(f"{tool.name}: {exc}") from exc

    latency_ms = int((time.perf_counter() - started) * 1000)
    if latency_ms > tool.max_runtime_ms:
        raise ToolTimeoutError(
            f"{tool.name} exceeded max_runtime_ms ({latency_ms}ms > {tool.max_runtime_ms}ms)"
        )
    return output, latency_ms


def list_tool_specs() -> list[dict[str, Any]]:
    """Return tool specs in a function-calling-shaped descriptor."""
    out: list[dict[str, Any]] = []
    for tool in REGISTRY.values():
        out.append(
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_model.model_json_schema(),
            }
        )
    return out


def get_tool(name: str) -> Tool | None:
    return REGISTRY.get(name)


ProviderCall = Callable[[str], str]
