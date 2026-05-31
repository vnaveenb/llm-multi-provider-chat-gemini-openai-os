"""Demo tool registry for the /tools endpoint."""

from __future__ import annotations

import ast
import datetime
import operator
from collections.abc import Callable
from typing import Any

TOOL_REGISTRY: dict[str, dict[str, Any]] = {}

_BINOPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNOPS: dict[type, Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.expr) -> float | int:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def register_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
) -> Callable[[Callable[..., str]], Callable[..., str]]:
    def decorator(func: Callable[..., str]) -> Callable[..., str]:
        TOOL_REGISTRY[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "callable": func,
        }
        return func

    return decorator


@register_tool(
    name="get_current_time",
    description="Returns the current UTC date and time as an ISO 8601 string.",
    parameters={"type": "object", "properties": {}, "required": []},
)
def get_current_time() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


@register_tool(
    name="calculate",
    description=(
        "Evaluates a safe arithmetic expression and returns the result as a string. "
        "Supports +, -, *, /, ** (or ^), % (or mod), //. "
        "Examples: '2 + 3', '2 ^ 10', '10 mod 3', '-5 * 2'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic expression to evaluate.",
            }
        },
        "required": ["expression"],
    },
)
def calculate(expression: str) -> str:
    expr = expression.replace("^", "**").replace(" mod ", " % ")
    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval_node(tree.body)
        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero"
    except (ValueError, TypeError) as exc:
        return f"Error: {exc}"
    except SyntaxError:
        return "Error: invalid expression"


def get_tool_schemas(names: list[str]) -> list[dict[str, Any]]:
    """Return LangChain-compatible tool schema dicts for the requested tool names."""
    schemas = []
    for name in names:
        entry = TOOL_REGISTRY.get(name)
        if entry is None:
            raise ValueError(f"Unknown tool: {name!r}")
        schemas.append(
            {
                "name": entry["name"],
                "description": entry["description"],
                "parameters": entry["parameters"],
            }
        )
    return schemas


def execute_tool(name: str, args: dict[str, Any]) -> str:
    """Execute a registered tool by name and return its string result."""
    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        raise ValueError(f"Unknown tool: {name!r}")
    try:
        return entry["callable"](**args)
    except Exception as exc:
        return f"Tool error: {exc}"


__all__ = ["TOOL_REGISTRY", "get_tool_schemas", "execute_tool"]
