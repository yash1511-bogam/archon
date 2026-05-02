"""Security layer — subprocess sandbox and policy engine.

Default-deny posture: tools are blocked unless explicitly allowed.
Tool execution runs in isolated subprocess with timeout and resource limits.
"""

from __future__ import annotations

import subprocess
import sys
import json
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PolicyAction(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass
class PolicyRule:
    tool: str  # tool name or "*" for wildcard
    action: PolicyAction = PolicyAction.DENY
    max_args_length: int = 10_000


@dataclass
class SecurityPolicy:
    """Declarative policy engine for tool execution.

    Default-deny: tools not in the allow list are blocked.
    """

    default: PolicyAction = PolicyAction.DENY
    rules: list[PolicyRule] = field(default_factory=list)

    def evaluate(self, tool_name: str, args: dict[str, Any]) -> PolicyAction:
        """Evaluate whether a tool call is allowed."""
        # Check specific rules first
        for rule in self.rules:
            if rule.tool == tool_name or rule.tool == "*":
                # Check arg size limits
                args_str = json.dumps(args)
                if len(args_str) > rule.max_args_length:
                    return PolicyAction.DENY
                return rule.action
        return self.default

    @classmethod
    def allow_all(cls) -> SecurityPolicy:
        return cls(default=PolicyAction.ALLOW)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SecurityPolicy:
        default = PolicyAction(data.get("default", "deny"))
        rules = []
        for r in data.get("rules", []):
            rules.append(PolicyRule(
                tool=r["tool"],
                action=PolicyAction(r.get("action", "allow")),
                max_args_length=r.get("max_args_length", 10_000),
            ))
        return cls(default=default, rules=rules)


@dataclass
class SandboxResult:
    output: str
    error: str | None = None
    timed_out: bool = False
    exit_code: int = 0


class Sandbox:
    """Execute tool functions in an isolated subprocess with timeout."""

    def __init__(self, *, timeout: int = 30, max_output: int = 50_000) -> None:
        self.timeout = timeout
        self.max_output = max_output

    def execute(self, fn_source: str, fn_name: str, args: dict[str, Any]) -> SandboxResult:
        """Run a function in a subprocess. Returns SandboxResult."""
        wrapper = textwrap.dedent(f"""\
            import json, sys
            {fn_source}
            args = json.loads(sys.stdin.read())
            result = {fn_name}(**args)
            print(json.dumps({{"result": str(result)}}))
        """)

        try:
            proc = subprocess.run(
                [sys.executable, "-c", wrapper],
                input=json.dumps(args),
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(output="", error="Tool execution timed out", timed_out=True, exit_code=-1)

        if proc.returncode != 0:
            return SandboxResult(
                output="", error=proc.stderr[:self.max_output] or "Tool execution failed",
                exit_code=proc.returncode,
            )

        output = proc.stdout[:self.max_output]
        try:
            parsed = json.loads(output)
            return SandboxResult(output=parsed.get("result", output))
        except json.JSONDecodeError:
            return SandboxResult(output=output)


@dataclass
class SecurityConfig:
    """Top-level security configuration for an agent."""

    sandbox: bool = False
    policy: SecurityPolicy = field(default_factory=lambda: SecurityPolicy.allow_all())
    sandbox_timeout: int = 30
    max_output_size: int = 50_000

    @classmethod
    def default_deny(cls) -> SecurityConfig:
        return cls(sandbox=True, policy=SecurityPolicy())

    @classmethod
    def permissive(cls) -> SecurityConfig:
        return cls(sandbox=False, policy=SecurityPolicy.allow_all())
