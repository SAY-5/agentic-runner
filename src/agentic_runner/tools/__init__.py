"""Tool registry — auto-imports each module to trigger @register_tool."""

from agentic_runner.tools import (  # noqa: F401  -- registers all tools
    calculate,
    composed,  # noqa: F401  -- registers composed tools
    extract_json,
    finish,
    http_get,
    query_db,
    read_file,
    summarize,
    write_file,
)
from agentic_runner.tools._base import (
    REGISTRY,
    Tool,
    ToolInvocationError,
    ToolTimeoutError,
    register_tool,
)
from agentic_runner.tools.composed import ComposedStep, ComposedTool

__all__ = [
    "REGISTRY",
    "ComposedStep",
    "ComposedTool",
    "Tool",
    "ToolInvocationError",
    "ToolTimeoutError",
    "register_tool",
]
