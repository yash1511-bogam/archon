"""Continuous evaluation engine — inline, async, and regression detection.

Three layers that run at different points in the agent lifecycle:

  1. **Inline validators** — run on the critical path, every request.
     Catch structural failures (schema, loops, budget) before they propagate.

  2. **Async quality scorers** — run off the critical path on sampled traffic.
     Catch semantic failures (task completion, coherence, tool efficiency).

  3. **Regression detectors** — run on population-level aggregates.
     Catch distribution shifts, cost anomalies, and failure rate changes.
"""

from __future__ import annotations

import re
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from archon.types import AgentResult, Step


# ── Evaluation result ──────────────────────────────────

class EvalSeverity(str, Enum):
    """How serious is the evaluation finding."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass
class EvalResult:
    """Result of a single evaluation check.

    Attributes:
        name: Evaluator name (e.g., "schema_validator", "loop_detector").
        severity: Pass, warning, or fail.
        message: Human-readable explanation.
        score: Numeric score (0.0–1.0) where applicable.
        metadata: Additional structured data for debugging.
    """

    name: str
    severity: EvalSeverity
    message: str
    score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Layer 1: Inline validators ─────────────────────────

class InlineValidator(ABC):
    """Base class for validators that run on every request."""

    @abstractmethod
    def validate(self, result: AgentResult) -> EvalResult: ...


class SchemaValidator(InlineValidator):
    """Check that the agent produced non-empty output."""

    def validate(self, result: AgentResult) -> EvalResult:
        if not result.output or not result.output.strip():
            return EvalResult("schema_validator", EvalSeverity.FAIL, "Empty output")
        if result.output.startswith("[Budget exceeded"):
            return EvalResult("schema_validator", EvalSeverity.WARNING, "Budget exceeded before completion")
        return EvalResult("schema_validator", EvalSeverity.PASS, "Output is non-empty")


class LoopDetector(InlineValidator):
    """Detect repeated tool calls that indicate the agent is stuck.

    Flags when the same tool is called 3+ times consecutively.
    """

    CONSECUTIVE_THRESHOLD = 3

    def validate(self, result: AgentResult) -> EvalResult:
        tool_calls = [s.tool_call for s in result.steps if s.tool_call]
        if len(tool_calls) < self.CONSECUTIVE_THRESHOLD:
            return EvalResult("loop_detector", EvalSeverity.PASS, "No loops detected")

        # Check for consecutive identical tool calls
        consecutive = 1
        for i in range(1, len(tool_calls)):
            if tool_calls[i] == tool_calls[i - 1]:
                consecutive += 1
                if consecutive >= self.CONSECUTIVE_THRESHOLD:
                    return EvalResult(
                        "loop_detector", EvalSeverity.FAIL,
                        f"Tool '{tool_calls[i]}' called {consecutive}x consecutively",
                        score=0.0,
                        metadata={"tool": tool_calls[i], "count": consecutive},
                    )
            else:
                consecutive = 1

        return EvalResult("loop_detector", EvalSeverity.PASS, "No loops detected")


class CostGuard(InlineValidator):
    """Flag runs that cost more than expected for their step count."""

    MAX_COST_PER_STEP = 0.50  # USD

    def validate(self, result: AgentResult) -> EvalResult:
        if result.step_count == 0:
            return EvalResult("cost_guard", EvalSeverity.PASS, "No steps")

        cost_per_step = result.total_cost_usd / result.step_count
        if cost_per_step > self.MAX_COST_PER_STEP:
            return EvalResult(
                "cost_guard", EvalSeverity.WARNING,
                f"High cost per step: ${cost_per_step:.4f}",
                score=max(0.0, 1.0 - (cost_per_step / self.MAX_COST_PER_STEP)),
            )
        return EvalResult("cost_guard", EvalSeverity.PASS, f"${cost_per_step:.4f}/step")


class ToolEfficiencyValidator(InlineValidator):
    """Flag runs where tool calls exceed 3× the step count (planning failure)."""

    EFFICIENCY_RATIO = 3.0

    def validate(self, result: AgentResult) -> EvalResult:
        tool_calls = sum(1 for s in result.steps if s.tool_call)
        if result.step_count == 0 or tool_calls == 0:
            return EvalResult("tool_efficiency", EvalSeverity.PASS, "No tool calls")

        ratio = tool_calls / result.step_count
        if ratio > self.EFFICIENCY_RATIO:
            return EvalResult(
                "tool_efficiency", EvalSeverity.WARNING,
                f"High tool call ratio: {ratio:.1f}x (possible planning failure)",
                score=max(0.0, 1.0 - (ratio / (self.EFFICIENCY_RATIO * 2))),
            )
        return EvalResult("tool_efficiency", EvalSeverity.PASS, f"Ratio: {ratio:.1f}x")


# ── Layer 2: Async quality scorers ─────────────────────

class AsyncScorer(ABC):
    """Base class for quality scorers that run off the critical path."""

    @abstractmethod
    async def score(self, prompt: str, result: AgentResult) -> EvalResult: ...


class OutputLengthScorer(AsyncScorer):
    """Score based on output length relative to prompt complexity.

    Very short outputs for complex prompts suggest incomplete answers.
    """

    MIN_OUTPUT_CHARS = 20

    async def score(self, prompt: str, result: AgentResult) -> EvalResult:
        output_len = len(result.output)
        if output_len < self.MIN_OUTPUT_CHARS:
            return EvalResult(
                "output_length", EvalSeverity.WARNING,
                f"Very short output ({output_len} chars)",
                score=output_len / self.MIN_OUTPUT_CHARS,
            )
        return EvalResult("output_length", EvalSeverity.PASS, f"{output_len} chars", score=1.0)


class CoherenceScorer(AsyncScorer):
    """Check that the output doesn't contain obvious contradictions or errors."""

    ERROR_PATTERNS = [
        re.compile(r"\[Tool error:", re.IGNORECASE),
        re.compile(r"\[BLOCKED by policy", re.IGNORECASE),
        re.compile(r"\[Unknown tool:", re.IGNORECASE),
        re.compile(r"\[REQUIRES APPROVAL", re.IGNORECASE),
    ]

    async def score(self, prompt: str, result: AgentResult) -> EvalResult:
        error_count = sum(
            1 for pattern in self.ERROR_PATTERNS if pattern.search(result.output)
        )
        if error_count > 0:
            return EvalResult(
                "coherence", EvalSeverity.WARNING,
                f"Output contains {error_count} error marker(s)",
                score=max(0.0, 1.0 - (error_count * 0.3)),
            )
        return EvalResult("coherence", EvalSeverity.PASS, "No error markers", score=1.0)


# ── Layer 3: Regression detectors ──────────────────────

@dataclass
class RegressionWindow:
    """A sliding window of recent evaluation scores for regression detection.

    Attributes:
        scores: Recent scores (0.0–1.0) for a specific metric.
        costs: Recent costs per run.
        failures: Per-entry failure flags, aligned with ``scores``/``costs``.
        window_size: Maximum entries to keep.
    """

    scores: list[float] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)
    failures: list[bool] = field(default_factory=list)
    window_size: int = 100

    def add(self, score: float, cost: float, failed: bool) -> None:
        """Add a data point to the window, trimming all aligned lists together."""
        self.scores.append(score)
        self.costs.append(cost)
        self.failures.append(failed)
        # Trim to window size. Keep scores, costs, and failures aligned so
        # ``failure_count`` reflects only the entries currently in the window.
        if len(self.scores) > self.window_size:
            self.scores = self.scores[-self.window_size:]
            self.costs = self.costs[-self.window_size:]
            self.failures = self.failures[-self.window_size:]

    @property
    def failure_count(self) -> int:
        """Number of failures currently within the window."""
        return sum(1 for f in self.failures if f)


class RegressionDetector:
    """Detect population-level regressions in agent quality and cost.

    Compares recent performance against a baseline to catch:
      - Score distribution shifts (quality degradation)
      - Cost anomalies (sudden spending spikes)
      - Failure rate increases
    """

    SCORE_DROP_THRESHOLD = 0.10  # 10% drop triggers warning
    COST_SPIKE_THRESHOLD = 2.0   # 2× cost increase triggers warning
    FAILURE_RATE_THRESHOLD = 0.15  # 15% failure rate triggers warning

    def __init__(self) -> None:
        self.windows: dict[str, RegressionWindow] = {}

    def record(self, agent_name: str, score: float, cost: float, failed: bool) -> None:
        """Record a data point for an agent."""
        if agent_name not in self.windows:
            self.windows[agent_name] = RegressionWindow()
        self.windows[agent_name].add(score, cost, failed)

    def check(self, agent_name: str) -> list[EvalResult]:
        """Check for regressions in the given agent's recent performance."""
        window = self.windows.get(agent_name)
        if not window or len(window.scores) < 10:
            return [EvalResult("regression", EvalSeverity.PASS, "Insufficient data")]

        results: list[EvalResult] = []

        # Score regression: compare last 10 vs previous 10
        recent = window.scores[-10:]
        baseline = window.scores[-20:-10] if len(window.scores) >= 20 else window.scores[:10]

        recent_mean = statistics.mean(recent)
        baseline_mean = statistics.mean(baseline)
        score_drop = baseline_mean - recent_mean

        if score_drop > self.SCORE_DROP_THRESHOLD:
            results.append(EvalResult(
                "score_regression", EvalSeverity.WARNING,
                f"Score dropped {score_drop:.2f} (baseline={baseline_mean:.2f} → recent={recent_mean:.2f})",
                score=recent_mean,
            ))
        else:
            results.append(EvalResult("score_regression", EvalSeverity.PASS, f"Score stable at {recent_mean:.2f}"))

        # Cost anomaly
        recent_cost = statistics.mean(window.costs[-10:])
        baseline_cost = statistics.mean(window.costs[-20:-10]) if len(window.costs) >= 20 else statistics.mean(window.costs[:10])

        if baseline_cost > 0 and recent_cost / baseline_cost > self.COST_SPIKE_THRESHOLD:
            results.append(EvalResult(
                "cost_anomaly", EvalSeverity.WARNING,
                f"Cost spike: ${baseline_cost:.4f} → ${recent_cost:.4f} ({recent_cost/baseline_cost:.1f}×)",
            ))
        else:
            results.append(EvalResult("cost_anomaly", EvalSeverity.PASS, f"Cost stable at ${recent_cost:.4f}"))

        # Failure rate
        total = len(window.scores)
        failure_rate = window.failure_count / total if total > 0 else 0
        if failure_rate > self.FAILURE_RATE_THRESHOLD:
            results.append(EvalResult(
                "failure_rate", EvalSeverity.WARNING,
                f"Failure rate: {failure_rate:.1%} ({window.failure_count}/{total})",
            ))
        else:
            results.append(EvalResult("failure_rate", EvalSeverity.PASS, f"Failure rate: {failure_rate:.1%}"))

        return results


# ── Evaluation engine ──────────────────────────────────

class EvalEngine:
    """Orchestrates all three evaluation layers.

    Usage::

        engine = EvalEngine()
        # After every agent run:
        inline_results = engine.run_inline(result)
        # Periodically:
        async_results = await engine.run_async(prompt, result)
        # On schedule:
        regression_results = engine.check_regressions("my_agent")
    """

    def __init__(
        self,
        *,
        inline: list[InlineValidator] | None = None,
        async_scorers: list[AsyncScorer] | None = None,
    ) -> None:
        self.inline_validators = inline or [
            SchemaValidator(),
            LoopDetector(),
            CostGuard(),
            ToolEfficiencyValidator(),
        ]
        self.async_scorers = async_scorers or [
            OutputLengthScorer(),
            CoherenceScorer(),
        ]
        self.regression = RegressionDetector()

    def run_inline(self, result: AgentResult) -> list[EvalResult]:
        """Run all inline validators (synchronous, on critical path)."""
        return [v.validate(result) for v in self.inline_validators]

    async def run_async(self, prompt: str, result: AgentResult) -> list[EvalResult]:
        """Run all async quality scorers (off critical path)."""
        return [await s.score(prompt, result) for s in self.async_scorers]

    def record_for_regression(
        self, agent_name: str, result: AgentResult, eval_results: list[EvalResult],
    ) -> None:
        """Record a run's evaluation results for regression tracking."""
        avg_score = statistics.mean([r.score for r in eval_results]) if eval_results else 1.0
        failed = any(r.severity == EvalSeverity.FAIL for r in eval_results)
        self.regression.record(agent_name, avg_score, result.total_cost_usd, failed)

    def check_regressions(self, agent_name: str) -> list[EvalResult]:
        """Check for population-level regressions."""
        return self.regression.check(agent_name)
