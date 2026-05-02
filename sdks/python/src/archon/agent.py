"""Agent — the core Archon API with full production harness."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
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

_DEFAULT_TRACE_DB = Path.home() / ".archon" / "traces.db"


class Agent:
    """An AI agent with built-in cost control, routing, security, and tracing.

    Every LLM call passes through 5 gates:
    1. Policy check — is this tool call allowed?
    2. Model routing — cheapest sufficient model
    3. Execute — LLM call + sandboxed tool execution
    4. Validate — sanitize tool outputs
    5. Trace — log everything
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
        """Execute the agent loop with full production harness."""
        run_id = str(uuid.uuid4())
        result = AgentResult(run_id=run_id, output="", started_at=datetime.now(timezone.utc))

        # Register run in trace store
        if self.trace_store:
            self.trace_store.start_run(run_id, self.name, prompt)
            self.trace_store.audit(run_id, self.name, "run_started", prompt[:200])

        # Gate 0: Check semantic cache
        if self.cache:
            routing = self._resolve_model(prompt)
            cached = self.cache.get(prompt, routing.model)
            if cached:
                result.output = cached.response
                result.finished_at = datetime.now(timezone.utc)
                step = Step(
                    id=f"{run_id}-cache", model=routing.model, tier=routing.tier,
                    cost_usd=0.0, cached=True,
                )
                result.steps.append(step)
                if self.trace_store:
                    self.trace_store.record_step(self.name, run_id, step)
                    self.trace_store.audit(run_id, self.name, "cache_hit")
                    self.trace_store.finish_run(run_id, result.output, 0.0, 1, 0, 0)
                return result

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]
        tool_schemas = [t.to_schema() for t in self.tools]
        tool_map = {t.name: t for t in self.tools}

        for step_num in range(self.max_steps):
            # Gate 1: Route to optimal model
            last_content = messages[-1].get("content", "") if messages else prompt
            routing = self._resolve_model(last_content if step_num > 0 else prompt)

            # Gate 2: Check budget
            try:
                self.budget.check(0.01)
            except BudgetExceeded:
                result.output = f"[Budget exceeded after {step_num} steps, spent ${self.budget.spent:.4f}]"
                if self.trace_store:
                    self.trace_store.audit(run_id, self.name, "budget_exceeded", f"${self.budget.spent:.4f}")
                break

            # Gate 3: Call LLM
            t0 = time.monotonic()
            response = await litellm.acompletion(
                model=routing.model,
                messages=messages,
                tools=tool_schemas if tool_schemas else None,
            )
            latency_ms = int((time.monotonic() - t0) * 1000)

            # Track cost
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            cost = 0.0
            if hasattr(response, "_hidden_params"):
                cost = response._hidden_params.get("response_cost", 0.0)
            if cost == 0.0:
                try:
                    cost = litellm.completion_cost(completion_response=response)
                except Exception:
                    cost = 0.0
            self.budget.record(cost)

            step = Step(
                id=f"{run_id}-{step_num}", model=routing.model, tier=routing.tier,
                input_tokens=input_tokens, output_tokens=output_tokens,
                cost_usd=cost, latency_ms=latency_ms,
            )

            choice = response.choices[0]
            msg = choice.message

            if msg.tool_calls:
                messages.append(msg.model_dump())
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    step.tool_call = fn_name

                    # Gate 1 (per tool): Policy check
                    args = json.loads(tc.function.arguments)
                    policy_result = self.security.policy.evaluate(fn_name, args)

                    if policy_result == PolicyAction.DENY:
                        tool_output = f"[BLOCKED by policy: {fn_name}]"
                        if self.trace_store:
                            self.trace_store.audit(run_id, self.name, "tool_denied", fn_name)
                    elif policy_result == PolicyAction.REQUIRE_APPROVAL:
                        tool_output = f"[REQUIRES APPROVAL: {fn_name}]"
                        if self.trace_store:
                            self.trace_store.audit(run_id, self.name, "tool_approval_required", fn_name)
                    elif fn_name in tool_map:
                        # Execute tool
                        try:
                            raw_result = tool_map[fn_name].fn(**args)
                            tool_output = str(raw_result)
                        except Exception as e:
                            tool_output = f"[Tool error: {e}]"
                            if self.trace_store:
                                self.trace_store.audit(run_id, self.name, "tool_error", f"{fn_name}: {e}")
                    else:
                        tool_output = f"[Unknown tool: {fn_name}]"

                    # Gate 4: Sanitize tool output
                    if self._sanitizer:
                        san = self._sanitizer.sanitize(tool_output)
                        if san.was_modified and self.trace_store:
                            self.trace_store.audit(
                                run_id, self.name, "output_sanitized",
                                f"{fn_name}: threats={san.threats_found} level={san.threat_level.value}",
                            )
                        tool_output = san.sanitized

                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": tool_output})

                # Gate 5: Record step
                self._record_step(result, step, run_id)
                continue

            # Final response — no tool calls
            self._record_step(result, step, run_id)
            result.output = msg.content or ""

            # Cache the result
            if self.cache and result.output:
                self.cache.put(prompt, routing.model, result.output, result.total_cost_usd)

            break

        result.finished_at = datetime.now(timezone.utc)
        result.model_usage = {}
        for s in result.steps:
            result.model_usage[s.model] = result.model_usage.get(s.model, 0) + 1

        if self.trace_store:
            self.trace_store.finish_run(
                run_id, result.output, result.total_cost_usd,
                len(result.steps), result.total_input_tokens + result.total_output_tokens,
                result.total_latency_ms,
            )

        return result

    def _record_step(self, result: AgentResult, step: Step, run_id: str) -> None:
        result.steps.append(step)
        result.total_cost_usd += step.cost_usd
        result.total_input_tokens += step.input_tokens
        result.total_output_tokens += step.output_tokens
        result.total_latency_ms += step.latency_ms
        if self.trace_store:
            self.trace_store.record_step(self.name, run_id, step)

    def _resolve_model(self, text: str) -> RoutingDecision:
        if self.model != "auto":
            return RoutingDecision(tier=Tier.STANDARD, model=self.model, reason="explicit model")
        return self._router.route(text, self.budget.remaining)
