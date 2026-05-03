"""Tool decorator for defining agent tools.

Usage::

    @tool
    def search(query: str) -> str:
        \"\"\"Search the web for information.\"\"\"
        return web_search(query)

The decorator inspects type hints to generate a JSON Schema
compatible with OpenAI/Anthropic/Google function-calling APIs.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, get_type_hints

# Python type → JSON Schema type mapping
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


@dataclass
class ToolDef:
    """A tool that an agent can invoke.

    Attributes:
        name: Function name, used as the tool identifier in LLM calls.
        description: Docstring, sent to the LLM to explain what the tool does.
        fn: The actual callable to execute.
        parameters: JSON Schema properties derived from type hints.
    """

    name: str
    description: str
    fn: Callable[..., Any]
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_schema(self) -> dict[str, Any]:
        """Convert to the OpenAI-compatible function-calling schema."""
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
    """Decorator that registers a function as an agent tool.

    Reads type hints to build a JSON Schema automatically.
    The function's docstring becomes the tool description sent to the LLM.
    """
    hints = get_type_hints(fn)
    signature = inspect.signature(fn)

    parameters: dict[str, Any] = {}
    for param_name in signature.parameters:
        if param_name == "return":
            continue
        python_type = hints.get(param_name, str)
        json_type = _TYPE_MAP.get(python_type, "string")
        parameters[param_name] = {"type": json_type}

    return ToolDef(
        name=fn.__name__,
        description=(fn.__doc__ or "").strip(),
        fn=fn,
        parameters=parameters,
    )
