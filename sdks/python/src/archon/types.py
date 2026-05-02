"""Core types shared across all Archon modules.

These are the data contracts that flow through the entire system:
Agent → Router → Budget → TraceStore → CLI.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import ClassVar

from pydantic import BaseModel, Field


class Tier(str, Enum):
    """Complexity tier for model routing.

    The router classifies every input into one of three tiers,
    then selects the cheapest model capable of handling that tier.
    """

    SIMPLE = "simple"
    STANDARD = "standard"
    COMPLEX = "complex"


class Step(BaseModel):
    """A single step in an agent execution trace.

    Every LLM call produces one Step. Tool calls, cache hits,
    and policy blocks are all recorded as steps for full auditability.
    """

    id: str
    model: str
    tier: Tier
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    tool_call: str | None = None
    cached: bool = False
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentResult(BaseModel):
    """Result of a complete agent run.

    Always includes cost, step count, and a trace URL — observability
    is not optional in Archon.
    """

    run_id: str
    output: str
    steps: list[Step] = Field(default_factory=list)
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: int = 0
    model_usage: dict[str, int] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None

    # Base URL for the dashboard; override to point trace_url at a custom host.
    _dashboard_base: ClassVar[str] = "http://localhost:8080"

    @property
    def cost(self) -> float:
        """Total cost in USD for this run."""
        return self.total_cost_usd

    @property
    def step_count(self) -> int:
        """Number of steps (LLM calls + tool calls) in this run."""
        return len(self.steps)

    @property
    def trace_url(self) -> str:
        """URL to view the full execution trace in the dashboard."""
        return f"{self._dashboard_base}/traces/{self.run_id}"
