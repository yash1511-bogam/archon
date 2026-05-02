"""Agent — the core Archon API.

Every LLM call passes through 5 gates:
  1. Policy check — is this tool call allowed?
  2. Model routing — cheapest sufficient model
  3. Execute — LLM call + tool execution
  4. Validate — sanitize tool outputs
  5. Trace — log everything

Usage::

    agent = Agent(
        name="researcher",
        instructions="Find accurate information.",
        tools=[search_web],
        model="auto",
        budget=Budget(max_per_run=0.50),
    )
    result = await agent.run("What caused the 2008 financial crisis?")
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import litellm

from archon.budget import Budget, BudgetExceeded
from archon.cache import SemanticCache
from archon.router import Router, RoutingDecision
from archon.sanitize import Sanitizer
from archon.security import PolicyAction, SecurityConfig
from archon.tool import ToolDef
from archon.trace import TraceStore
from archon.types import AgentResult, Step, Tier

# Conservative cost estimate for budget pre-check (USD).
# Used before the actual LLM call to avoid overspending.
ESTIMATED_COST_PER_CALL = 0.01


class Agent:
    """An AI agent with built-in cost control, routing, security, and tracing.

    Args:
        name: Human-readable agent name (used in traces and audit logs).
        instructions: System prompt — the agent's personality and rules.
        tools: List of tools the agent can invoke.
        model: LLM model name, or ``"auto"`` for automatic routing.
        budget: Budget with hard spending limits.
        max_steps: Maximum LLM calls before the agent stops.
        security: Security policy and sandbox configuration.
        cache: Semantic cache to skip repeated queries.
        trace_store: SQLite trace store for observability.
        sanitize: If True, sanitize tool outputs before passing to LLM.
    """

    def __init__(
        self,
        *,
        name: str,
        instructions: str,
        tools: list[ToolDef] | None = None,
        model: str = "auto",
        budget: Budget | None = None,
        max_steps: int = 25,
        security: SecurityConfig | None = None,
        cache: SemanticCache | None = None,
        trace_store: TraceStore | None = None,
        sanitize: bool = True,
    ) -> None:
        self.name = name
        self.instructions = instructions
        self.tools = tools or []
        self.model = model
        self.budget = budget or Budget()
        self.max_steps = max_steps
        self.security = security or SecurityConfig.permissive()
        self.cache = cache
        self.trace_store = trace_store
        self._sanitizer = Sanitizer(strict=sanitize) if sanitize else None
        self._router = Router()

    async def run(self, prompt: str) -> AgentResult:
        """Execute the agent loop with the full production harness.

        Returns:
            AgentResult with output, cost, steps, and trace URL.
        """
        run_id = str(uuid.uuid4())
        result = AgentResult(run_id=run_id, output="", started_at=datetime.now(timezone.utc))

        self._audit(run_id, "run_started", prompt[:200])

        # Gate 0: Check semantic cache before any LLM call
        cached_result = self._check_cache(prompt, run_id, result)
        if cached_result:
            return cached_result

        # Build conversation and run the agent loop
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]
        tool_schemas = [t.to_schema() for t in self.tools]
        tool_map = {t.name: t for t in self.tools}

        for step_number in range(self.max_steps):
            # Gate 1: Route to optimal model
            current_text = prompt if step_number == 0 else self._last_content(messages)
            routing = self._resolve_model(current_text)

            # Gate 2: Check budget before calling LLM
            if not self._check_budget(run_id, step_number, result):
                break

            # Gate 3: Call LLM
            step, response_message = await self._call_llm(
                routing, messages, tool_schemas, run_id, step_number,
            )

            # Handle tool calls or final response
            if response_message.tool_calls:
                messages.append(response_message.model_dump())
                self._handle_tool_calls(
                    response_message.tool_calls, tool_map, messages, step, run_id,
                )
                self._record_step(result, step, run_id)
                continue

            # Final response — no tool calls
            self._record_step(result, step, run_id)
            result.output = response_message.content or ""
            self._cache_result(prompt, routing.model, result.output, result.total_cost_usd)
            break

        return self._finalize(result, run_id)

    # ── Gate 0: Cache ──────────────────────────────────

    def _check_cache(self, prompt: str, run_id: str, result: AgentResult) -> AgentResult | None:
        """Return a cached result if available, otherwise None."""
        if not self.cache:
            return None

        routing = self._resolve_model(prompt)
        cached = self.cache.get(prompt, routing.model)
        if not cached:
            return None

        result.output = cached.response
        result.finished_at = datetime.now(timezone.utc)
        cache_step = Step(id=f"{run_id}-cache", model=routing.model, tier=routing.tier, cached=True)
        result.steps.append(cache_step)

        if self.trace_store:
            self.trace_store.record_step(self.name, run_id, cache_step)
            self.trace_store.start_run(run_id, self.name, "")
            self.trace_store.finish_run(run_id, result.output, 0.0, 1, 0, 0)

        self._audit(run_id, "cache_hit")
        return result

    def _cache_result(self, prompt: str, model: str, output: str, cost: float) -> None:
        """Store a successful result in the cache."""
        if self.cache and output:
            self.cache.put(prompt, model, output, cost)

    # ── Gate 1: Routing ────────────────────────────────

    def _resolve_model(self, text: str) -> RoutingDecision:
        """Pick the model — explicit or auto-routed based on complexity."""
        if self.model != "auto":
            return RoutingDecision(tier=Tier.STANDARD, model=self.model, reason="explicit model")
        return self._router.route(text, self.budget.remaining)

    # ── Gate 2: Budget ─────────────────────────────────

    def _check_budget(self, run_id: str, step_number: int, result: AgentResult) -> bool:
        """Return True if budget allows another call, False to stop."""
        try:
            self.budget.check(ESTIMATED_COST_PER_CALL)
            return True
        except BudgetExceeded:
            result.output = (
                f"[Budget exceeded after {step_number} steps, spent ${self.budget.spent:.4f}]"
            )
            self._audit(run_id, "budget_exceeded", f"${self.budget.spent:.4f}")
            return False

    # ── Gate 3: LLM execution ──────────────────────────

    async def _call_llm(
        self,
        routing: RoutingDecision,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        run_id: str,
        step_number: int,
    ) -> tuple[Step, Any]:
        """Call the LLM and return a (Step, message) tuple."""
        start_time = time.monotonic()
        response = await litellm.acompletion(
            model=routing.model,
            messages=messages,
            tools=tool_schemas if tool_schemas else None,
        )
        latency_ms = int((time.monotonic() - start_time) * 1000)

        cost = self._extract_cost(response)
        self.budget.record(cost)

        usage = response.usage
        step = Step(
            id=f"{run_id}-{step_number}",
            model=routing.model,
            tier=routing.tier,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cost_usd=cost,
            latency_ms=latency_ms,
        )

        return step, response.choices[0].message

    # ── Gate 4: Tool execution + sanitization ──────────

    def _handle_tool_calls(
        self,
        tool_calls: list[Any],
        tool_map: dict[str, ToolDef],
        messages: list[dict[str, Any]],
        step: Step,
        run_id: str,
    ) -> None:
        """Execute each tool call with policy check and output sanitization."""
        for tool_call in tool_calls:
            fn_name = tool_call.function.name
            step.tool_call = fn_name
            args = json.loads(tool_call.function.arguments)

            # Policy check
            tool_output = self._execute_with_policy(fn_name, args, tool_map, run_id)

            # Sanitize output before passing back to LLM
            tool_output = self._sanitize_output(tool_output, fn_name, run_id)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_output,
            })

    def _execute_with_policy(
        self,
        fn_name: str,
        args: dict[str, Any],
        tool_map: dict[str, ToolDef],
        run_id: str,
    ) -> str:
        """Check policy, then execute the tool. Returns the tool output string."""
        policy_result = self.security.policy.evaluate(fn_name, args)

        if policy_result == PolicyAction.DENY:
            self._audit(run_id, "tool_denied", fn_name)
            return f"[BLOCKED by policy: {fn_name}]"

        if policy_result == PolicyAction.REQUIRE_APPROVAL:
            self._audit(run_id, "tool_approval_required", fn_name)
            return f"[REQUIRES APPROVAL: {fn_name}]"

        if fn_name not in tool_map:
            return f"[Unknown tool: {fn_name}]"

        try:
            raw_result = tool_map[fn_name].fn(**args)
            return str(raw_result)
        except Exception as exc:
            self._audit(run_id, "tool_error", f"{fn_name}: {exc}")
            return f"[Tool error: {exc}]"

    def _sanitize_output(self, output: str, fn_name: str, run_id: str) -> str:
        """Strip injection patterns from tool output."""
        if not self._sanitizer:
            return output

        result = self._sanitizer.sanitize(output)
        if result.was_modified:
            self._audit(
                run_id, "output_sanitized",
                f"{fn_name}: threats={result.threats_found} level={result.threat_level.value}",
            )
        return result.sanitized

    # ── Gate 5: Tracing ────────────────────────────────

    def _record_step(self, result: AgentResult, step: Step, run_id: str) -> None:
        """Add a step to the result and persist it to the trace store."""
        result.steps.append(step)
        result.total_cost_usd += step.cost_usd
        result.total_input_tokens += step.input_tokens
        result.total_output_tokens += step.output_tokens
        result.total_latency_ms += step.latency_ms

        if self.trace_store:
            self.trace_store.record_step(self.name, run_id, step)

    def _audit(self, run_id: str, action: str, detail: str | None = None) -> None:
        """Append to the immutable audit log."""
        if self.trace_store:
            self.trace_store.audit(run_id, self.name, action, detail)

    # ── Helpers ────────────────────────────────────────

    def _finalize(self, result: AgentResult, run_id: str) -> AgentResult:
        """Compute final totals and persist the run summary."""
        result.finished_at = datetime.now(timezone.utc)
        result.model_usage = {}
        for step in result.steps:
            result.model_usage[step.model] = result.model_usage.get(step.model, 0) + 1

        if self.trace_store:
            total_tokens = result.total_input_tokens + result.total_output_tokens
            self.trace_store.start_run(run_id, self.name, "")
            self.trace_store.finish_run(
                run_id, result.output, result.total_cost_usd,
                len(result.steps), total_tokens, result.total_latency_ms,
            )
        return result

    @staticmethod
    def _last_content(messages: list[dict[str, Any]]) -> str:
        """Extract the content of the last message for routing classification."""
        if messages:
            return messages[-1].get("content", "")
        return ""

    @staticmethod
    def _extract_cost(response: Any) -> float:
        """Extract cost from a LiteLLM response, with fallback."""
        cost = 0.0
        if hasattr(response, "_hidden_params"):
            cost = response._hidden_params.get("response_cost", 0.0)
        if cost == 0.0:
            try:
                cost = litellm.completion_cost(completion_response=response)
            except Exception:
                cost = 0.0
        return cost
