"""Security layer — subprocess sandbox and declarative policy engine.

Default-deny posture: tools are blocked unless explicitly allowed.
Tool execution can run in an isolated subprocess with timeout.

Usage::

    policy = SecurityPolicy(rules=[
        PolicyRule(tool="search_web", action=PolicyAction.ALLOW),
        PolicyRule(tool="send_email", action=PolicyAction.REQUIRE_APPROVAL),
    ])
    # Everything else is denied by default.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)

# ── Default limits ─────────────────────────────────────

DEFAULT_SANDBOX_TIMEOUT_SECONDS = 30
DEFAULT_MAX_OUTPUT_SIZE = 50_000
DEFAULT_MAX_ARGS_LENGTH = 10_000


class PolicyAction(StrEnum):
    """What to do when an agent tries to call a tool."""

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass
class PolicyRule:
    """A single rule in the security policy.

    Attributes:
        tool: Tool name to match, or ``"*"`` for wildcard.
        action: What to do when this tool is called.
        max_args_length: Maximum serialized argument size (bytes). Prevents payload stuffing.
    """

    tool: str
    action: PolicyAction = PolicyAction.DENY
    max_args_length: int = DEFAULT_MAX_ARGS_LENGTH


@dataclass
class SecurityPolicy:
    """Declarative policy engine for tool execution.

    Evaluates tool calls against a list of rules. If no rule matches,
    the default action applies (deny by default).
    """

    default: PolicyAction = PolicyAction.DENY
    rules: list[PolicyRule] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._warn_if_wildcard_shadows_rules()

    def _warn_if_wildcard_shadows_rules(self) -> None:
        """Emit a warning if a ``"*"`` wildcard rule precedes more specific rules.

        ``evaluate`` returns on first match, so a wildcard that appears
        before specific rules will always win and silently disable the
        more specific rules below it. That is almost never intended.
        """
        for i, rule in enumerate(self.rules):
            if rule.tool == "*":
                shadowed = [r.tool for r in self.rules[i + 1:] if r.tool != "*"]
                if shadowed:
                    logger.warning(
                        "SecurityPolicy: wildcard rule at index %d shadows %d "
                        "subsequent specific rule(s): %s. Move the wildcard to "
                        "the end of the rules list so specific rules can match first.",
                        i, len(shadowed), shadowed,
                    )
                # Only the first wildcard needs to be reported.
                return

    def evaluate(self, tool_name: str, args: dict[str, Any]) -> PolicyAction:
        """Decide whether a tool call should proceed.

        Checks rules in order. First match wins. Falls back to ``self.default``.
        """
        for rule in self.rules:
            if rule.tool == tool_name or rule.tool == "*":
                serialized_args = json.dumps(args)
                if len(serialized_args) > rule.max_args_length:
                    return PolicyAction.DENY
                return rule.action
        return self.default

    @classmethod
    def allow_all(cls) -> SecurityPolicy:
        """Create a permissive policy that allows all tool calls."""
        return cls(default=PolicyAction.ALLOW)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SecurityPolicy:
        """Create a policy from a dictionary (e.g., parsed YAML config)."""
        default = PolicyAction(data.get("default", "deny"))
        rules = [
            PolicyRule(
                tool=r["tool"],
                action=PolicyAction(r.get("action", "allow")),
                max_args_length=r.get("max_args_length", DEFAULT_MAX_ARGS_LENGTH),
            )
            for r in data.get("rules", [])
        ]
        return cls(default=default, rules=rules)


# ── Subprocess sandbox ─────────────────────────────────

@dataclass
class SandboxResult:
    """Result of executing a tool in a sandboxed subprocess."""

    output: str
    error: str | None = None
    timed_out: bool = False
    exit_code: int = 0


class Sandbox:
    """Execute tool functions in an isolated subprocess with timeout.

    The tool runs in a separate Python process with no shared memory.
    This prevents a compromised tool from accessing the agent's state.
    """

    def __init__(
        self,
        *,
        timeout_seconds: int = DEFAULT_SANDBOX_TIMEOUT_SECONDS,
        max_output_size: int = DEFAULT_MAX_OUTPUT_SIZE,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_output_size = max_output_size

    def execute(self, fn_source: str, fn_name: str, args: dict[str, Any]) -> SandboxResult:
        """Run a function in a subprocess. Returns structured result."""
        wrapper_code = textwrap.dedent(f"""\
            import json, sys
            {fn_source}
            args = json.loads(sys.stdin.read())
            result = {fn_name}(**args)
            print(json.dumps({{"result": str(result)}}))
        """)

        try:
            # Intentional: sandbox runs tool code in an isolated ``python -c``
            # subprocess with a timeout. Arguments are passed via stdin (not
            # argv) so there is no shell interpretation of untrusted data.
            process = subprocess.run(  # noqa: S603 — sandbox by design
                [sys.executable, "-c", wrapper_code],
                input=json.dumps(args),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                output="", error="Tool execution timed out",
                timed_out=True, exit_code=-1,
            )

        if process.returncode != 0:
            error_message = process.stderr[: self.max_output_size] or "Tool execution failed"
            return SandboxResult(output="", error=error_message, exit_code=process.returncode)

        raw_output = process.stdout[: self.max_output_size]
        try:
            parsed = json.loads(raw_output)
            return SandboxResult(output=parsed.get("result", raw_output))
        except json.JSONDecodeError:
            return SandboxResult(output=raw_output)


# ── Top-level configuration ───────────────────────────

@dataclass
class SecurityConfig:
    """Top-level security configuration for an agent.

    Attributes:
        sandbox: If True, tool calls execute in isolated subprocesses.
        policy: The policy engine that decides allow/deny/approval per tool.
        sandbox_timeout: Seconds before a sandboxed tool call is killed.
        max_output_size: Maximum characters returned from a tool call.
    """

    sandbox: bool = False
    policy: SecurityPolicy = field(default_factory=lambda: SecurityPolicy.allow_all())
    sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT_SECONDS
    max_output_size: int = DEFAULT_MAX_OUTPUT_SIZE

    @classmethod
    def default_deny(cls) -> SecurityConfig:
        """Secure defaults: sandbox enabled, all tools denied unless allowed."""
        return cls(sandbox=True, policy=SecurityPolicy())

    @classmethod
    def permissive(cls) -> SecurityConfig:
        """Development defaults: no sandbox, all tools allowed."""
        return cls(sandbox=False, policy=SecurityPolicy.allow_all())
