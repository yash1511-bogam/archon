"""Agent — the core Archon API."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import litellm

from archon.budget import Budget, BudgetExceeded
from archon.router import Router, RoutingDecision
from archon.tool import ToolDef
from archon.types import AgentResult, Step, Tier


class Agent:
    """An AI agent with built-in cost control, routing, and tracing."""

    def __init__(
        self,
        *,
        name: str,
        instructions: str,
        tools: list[ToolDef] | None = None,
        model: str = "auto",
        budget: Budget | None = None,
        max_steps: int = 25,
    ) -> None:
        self.name = name
        self.instructions = instructions
        self.tools = tools or []
        self.model = model
        self.budget = budget or Budget()
        self.max_steps = max_steps
        self._router = Router()

    async def run(self, prompt: str) -> AgentResult:
        """Execute the agent loop and return a result with cost tracking."""
        result = AgentResult(
            run_id=str(uuid.uuid4()),
            output="",
            started_at=datetime.now(timezone.utc),
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]

        tool_schemas = [t.to_schema() for t in self.tools]
        tool_map = {t.name: t for t in self.tools}

        for step_num in range(self.max_steps):
            # 1. Route to optimal model
            routing = self._resolve_model(prompt if step_num == 0 else messages[-1].get("content", ""))

            # 2. Check budget before calling
            estimated_cost = 0.01  # Conservative estimate
            try:
                self.budget.check(estimated_cost)
            except BudgetExceeded:
                result.output = f"[Budget exceeded after {step_num} steps, spent ${self.budget.spent:.4f}]"
                break

            # 3. Call LLM
            t0 = time.monotonic()
            response = await litellm.acompletion(
                model=routing.model,
                messages=messages,
                tools=tool_schemas if tool_schemas else None,
            )
            latency_ms = int((time.monotonic() - t0) * 1000)

            # 4. Track cost
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            cost = response._hidden_params.get("response_cost", 0.0) if hasattr(response, "_hidden_params") else 0.0
            if cost == 0.0:
                cost = litellm.completion_cost(completion_response=response)

            self.budget.record(cost)

            step = Step(
                id=f"{result.run_id}-{step_num}",
                model=routing.model,
                tier=routing.tier,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                latency_ms=latency_ms,
                cached=False,
                timestamp=datetime.now(timezone.utc),
            )

            # 5. Handle response
            choice = response.choices[0]
            msg = choice.message

            if msg.tool_calls:
                messages.append(msg.model_dump())
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    step.tool_call = fn_name
                    result.steps.append(step)
                    result.total_cost_usd += cost
                    result.total_input_tokens += input_tokens
                    result.total_output_tokens += output_tokens
                    result.total_latency_ms += latency_ms

                    # Execute tool
                    if fn_name in tool_map:
                        args = json.loads(tc.function.arguments)
                        tool_result = tool_map[fn_name].fn(**args)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": str(tool_result),
                        })
                continue

            # No tool calls — final response
            result.steps.append(step)
            result.total_cost_usd += cost
            result.total_input_tokens += input_tokens
            result.total_output_tokens += output_tokens
            result.total_latency_ms += latency_ms
            result.output = msg.content or ""
            break

        result.finished_at = datetime.now(timezone.utc)
        result.model_usage = {}
        for s in result.steps:
            result.model_usage[s.model] = result.model_usage.get(s.model, 0) + 1

        return result

    def _resolve_model(self, text: str) -> RoutingDecision:
        if self.model != "auto":
            return RoutingDecision(
                tier=Tier.STANDARD,
                model=self.model,
                reason="explicit model",
            )
        return self._router.route(text, self.budget.remaining)
