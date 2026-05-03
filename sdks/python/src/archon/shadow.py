"""Shadow deployments — run a new agent version in parallel and compare scores.

The shadow runner executes both the current (primary) and candidate (shadow)
agents against the same prompt, evaluates both, and returns a comparison.
The shadow agent's output is never served to users — only used for scoring.

Usage::

    shadow = ShadowRunner(primary=current_agent, candidate=new_agent)
    comparison = await shadow.run("user prompt")
    if comparison.candidate_is_better:
        promote(new_agent)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from archon.eval import EvalEngine, EvalResult, EvalSeverity
from archon.types import AgentResult


@runtime_checkable
class Runnable(Protocol):
    """Any object with an async ``run(prompt) -> AgentResult`` method."""

    name: str
    async def run(self, prompt: str) -> AgentResult: ...


@dataclass
class ShadowComparison:
    """Result of comparing primary and candidate agent runs.

    Attributes:
        primary_result: The primary agent's output (served to users).
        candidate_result: The shadow agent's output (never served).
        primary_scores: Evaluation results for the primary.
        candidate_scores: Evaluation results for the candidate.
        primary_avg_score: Average evaluation score for primary.
        candidate_avg_score: Average evaluation score for candidate.
        candidate_is_better: True if candidate scores higher with no regressions.
        cost_delta_usd: Cost difference (candidate - primary). Negative = cheaper.
        recommendation: Human-readable promotion recommendation.
    """

    primary_result: AgentResult
    candidate_result: AgentResult
    primary_scores: list[EvalResult] = field(default_factory=list)
    candidate_scores: list[EvalResult] = field(default_factory=list)
    primary_avg_score: float = 0.0
    candidate_avg_score: float = 0.0
    candidate_is_better: bool = False
    cost_delta_usd: float = 0.0
    recommendation: str = ""


class ShadowRunner:
    """Run two agent versions in parallel and compare their evaluation scores.

    The primary agent's output is always returned to the user.
    The candidate runs silently in the background for comparison only.

    Args:
        primary: The current production agent.
        candidate: The new agent version to evaluate.
        eval_engine: Evaluation engine for scoring both outputs.
        min_score_improvement: Minimum score improvement to recommend promotion.
    """

    MIN_SCORE_IMPROVEMENT = 0.0  # Candidate must be at least as good

    def __init__(
        self,
        *,
        primary: Runnable,
        candidate: Runnable,
        eval_engine: EvalEngine | None = None,
        min_score_improvement: float = 0.0,
    ) -> None:
        self.primary = primary
        self.candidate = candidate
        self.eval_engine = eval_engine or EvalEngine()
        self.min_score_improvement = min_score_improvement

    async def run(self, prompt: str) -> ShadowComparison:
        """Execute both agents in parallel and compare results.

        Returns:
            ShadowComparison with scores, cost delta, and promotion recommendation.
        """
        # Run both agents concurrently. Use return_exceptions=True so a
        # candidate crash never takes down the primary result.
        primary_result, candidate_result = await asyncio.gather(
            self.primary.run(prompt),
            self.candidate.run(prompt),
            return_exceptions=True,
        )

        # A primary failure must still propagate — the user's result depends on it.
        if isinstance(primary_result, BaseException):
            raise primary_result

        # A candidate failure is survivable: we record a failed AgentResult
        # instead of crashing the shadow comparison.
        if isinstance(candidate_result, BaseException):
            candidate_result = AgentResult(
                run_id="shadow-failed",
                output=f"[Shadow candidate error: {candidate_result}]",
            )

        # Evaluate both
        primary_inline = self.eval_engine.run_inline(primary_result)
        candidate_inline = self.eval_engine.run_inline(candidate_result)

        primary_async = await self.eval_engine.run_async(prompt, primary_result)
        candidate_async = await self.eval_engine.run_async(prompt, candidate_result)

        all_primary = primary_inline + primary_async
        all_candidate = candidate_inline + candidate_async

        primary_avg = _avg_score(all_primary)
        candidate_avg = _avg_score(all_candidate)

        cost_delta = candidate_result.total_cost_usd - primary_result.total_cost_usd
        candidate_has_failures = any(r.severity == EvalSeverity.FAIL for r in all_candidate)
        score_improvement = candidate_avg - primary_avg

        is_better = (
            not candidate_has_failures
            and score_improvement >= self.min_score_improvement
        )

        recommendation = _build_recommendation(
            primary_avg, candidate_avg, cost_delta, candidate_has_failures, is_better,
        )

        return ShadowComparison(
            primary_result=primary_result,
            candidate_result=candidate_result,
            primary_scores=all_primary,
            candidate_scores=all_candidate,
            primary_avg_score=primary_avg,
            candidate_avg_score=candidate_avg,
            candidate_is_better=is_better,
            cost_delta_usd=cost_delta,
            recommendation=recommendation,
        )

    async def run_batch(self, prompts: list[str]) -> list[ShadowComparison]:
        """Run shadow comparison on multiple prompts concurrently."""
        return await asyncio.gather(*[self.run(p) for p in prompts])


def _avg_score(results: list[EvalResult]) -> float:
    """Compute average score from evaluation results."""
    if not results:
        return 0.0
    return sum(r.score for r in results) / len(results)


def _build_recommendation(
    primary_avg: float,
    candidate_avg: float,
    cost_delta: float,
    has_failures: bool,
    is_better: bool,
) -> str:
    """Generate a human-readable promotion recommendation."""
    if has_failures:
        return "DO NOT PROMOTE — candidate has evaluation failures"

    delta = candidate_avg - primary_avg
    cost_str = f"${abs(cost_delta):.4f} {'cheaper' if cost_delta < 0 else 'more expensive'}"

    if is_better and delta > 0.05:
        return f"PROMOTE — candidate scores {delta:.2f} higher, {cost_str}"
    if is_better and cost_delta < 0:
        return f"PROMOTE — similar quality, {cost_str}"
    if is_better:
        return f"SAFE TO PROMOTE — no regressions detected, {cost_str}"
    return f"HOLD — candidate scores {delta:.2f} lower, {cost_str}"
