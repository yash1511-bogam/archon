"""Pattern-based model router. Zero LLM calls, zero latency."""

from __future__ import annotations

from dataclasses import dataclass, field

from archon.types import Tier

# Complexity signal keywords
_ANALYSIS = {"analyze", "compare", "evaluate", "architect", "design", "optimize", "debug", "refactor"}
_MATH = {"calculate", "prove", "derive", "equation", "algorithm", "complexity"}
_MULTI_STEP = {"step 1", "first,", "then,", "finally,", "after that", "next,"}

DEFAULT_MODELS: dict[Tier, list[str]] = {
    Tier.SIMPLE: ["gemini-2.5-flash", "gpt-4.1-nano"],
    Tier.STANDARD: ["claude-sonnet-4.6", "gpt-4.1-mini"],
    Tier.COMPLEX: ["claude-opus-4.6", "o4-mini"],
}


@dataclass
class RoutingDecision:
    tier: Tier
    model: str
    reason: str
    fallback: str | None = None


@dataclass
class Router:
    models: dict[Tier, list[str]] = field(default_factory=lambda: dict(DEFAULT_MODELS))

    def route(self, text: str, remaining_budget: float | None = None) -> RoutingDecision:
        tier = self._classify(text)

        # Budget-aware downgrade
        if remaining_budget is not None:
            if tier == Tier.COMPLEX and remaining_budget < 0.10:
                tier = Tier.STANDARD
            if tier == Tier.STANDARD and remaining_budget < 0.05:
                tier = Tier.SIMPLE

        candidates = self.models.get(tier, self.models[Tier.STANDARD])
        return RoutingDecision(
            tier=tier,
            model=candidates[0],
            reason=f"classified as {tier.value}",
            fallback=candidates[1] if len(candidates) > 1 else None,
        )

    def _classify(self, text: str) -> Tier:
        score = 0
        lower = text.lower()
        words = lower.split()

        if len(words) > 200:
            score += 3
        elif len(words) > 50:
            score += 1

        if "```" in text:
            score += 2
        if any(kw in text for kw in ("def ", "function ", "class ")):
            score += 2

        for kw in _MULTI_STEP:
            if kw in lower:
                score += 1
        for kw in _ANALYSIS:
            if kw in lower:
                score += 2
        for kw in _MATH:
            if kw in lower:
                score += 2

        if score <= 2:
            return Tier.SIMPLE
        if score <= 6:
            return Tier.STANDARD
        return Tier.COMPLEX
