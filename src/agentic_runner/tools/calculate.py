"""Safe arithmetic expression evaluator."""

from __future__ import annotations

import ast
import operator as op
from collections.abc import Callable
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from agentic_runner.tools._base import ToolInvocationError, register_tool


class CalculateInput(BaseModel):
    expression: str = Field(min_length=1, max_length=200)


class CalculateOutput(BaseModel):
    value: float


_BIN_OPS: dict[type[ast.AST], Callable[[float, float], float]] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
}

_UNARY_OPS: dict[type[ast.AST], Callable[[float], float]] = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise ToolInvocationError("calculate: only numeric literals are allowed")
        return float(node.value)
    if isinstance(node, ast.BinOp):
        bin_fn = _BIN_OPS.get(type(node.op))
        if bin_fn is None:
            raise ToolInvocationError(f"calculate: operator not allowed: {type(node.op).__name__}")
        return float(bin_fn(_eval(node.left), _eval(node.right)))
    if isinstance(node, ast.UnaryOp):
        un_fn = _UNARY_OPS.get(type(node.op))
        if un_fn is None:
            raise ToolInvocationError(f"calculate: unary op not allowed: {type(node.op).__name__}")
        return float(un_fn(_eval(node.operand)))
    raise ToolInvocationError(f"calculate: AST node not allowed: {type(node).__name__}")


@register_tool
class CalculateTool:
    name: ClassVar[str] = "calculate"
    description: ClassVar[str] = "Evaluate a safe arithmetic expression and return its value."
    input_model: ClassVar[type[BaseModel]] = CalculateInput
    output_model: ClassVar[type[BaseModel]] = CalculateOutput
    max_runtime_ms: ClassVar[int] = 100
    idempotent: ClassVar[bool] = True

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        parsed = CalculateInput.model_validate(args)
        try:
            tree = ast.parse(parsed.expression, mode="eval")
        except SyntaxError as exc:
            raise ToolInvocationError(f"calculate: invalid expression: {exc}") from exc
        value = _eval(tree)
        return {"value": value}
