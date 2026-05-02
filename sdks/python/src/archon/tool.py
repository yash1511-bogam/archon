"""Tool decorator for defining agent tools."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints


@dataclass
class ToolDef:
    """A tool definition that an agent can invoke."""

    name: str
    description: str
    fn: Callable[..., Any]
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_schema(self) -> dict[str, Any]:
        """Convert to LLM-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": list(self.parameters.keys()),
                },
            },
        }


def tool(fn: Callable[..., Any]) -> ToolDef:
    """Decorator to register a function as an agent tool."""
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)

    params: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name == "return":
            continue
        hint = hints.get(name, str)
        json_type = _python_type_to_json(hint)
        params[name] = {"type": json_type}

    return ToolDef(
        name=fn.__name__,
        description=(fn.__doc__ or "").strip(),
        fn=fn,
        parameters=params,
    )


def _python_type_to_json(t: type) -> str:
    mapping = {str: "string", int: "integer", float: "number", bool: "boolean"}
    return mapping.get(t, "string")
