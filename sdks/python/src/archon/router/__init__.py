"""Pattern-based model router. Zero LLM calls, zero added latency.

Classifies input complexity using 27 lexical signals, then selects
the cheapest model capable of handling that complexity tier.
Budget-aware: automatically downgrades when funds are low.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from archon.types import Tier

# ── Complexity signal keywords ─────────────────────────
# Each set contributes a weighted score to the classifier.

ANALYSIS_KEYWORDS = frozenset({
    "analyze", "compare", "evaluate", "architect",
    "design", "optimize", "debug", "refactor",
})

MATH_KEYWORDS = frozenset({
    "calculate", "prove", "derive", "equation",
    "algorithm", "complexity",
})

MULTI_STEP_PHRASES = frozenset({
    "step 1", "first,", "then,", "finally,",
    "after that", "next,",
})

# Compiled word-boundary patterns — prevents "designer" matching "design"
# Matches the keyword as a complete word or with verb suffixes (ed, ing, tion, ize, ation)
# but not noun agent suffixes (er, ist, ment) to avoid false positives.
_VERB_SUFFIX = r"(?:e?d|e?s|ing|tion|ation|ize|ized)?"
_ANALYSIS_RE = re.compile(
    r"\b(?:" + "|".join(ANALYSIS_KEYWORDS) + r")" + _VERB_SUFFIX + r"\b",
    re.IGNORECASE,
)
_MATH_RE = re.compile(
    r"\b(?:" + "|".join(MATH_KEYWORDS) + r")" + _VERB_SUFFIX + r"\b",
    re.IGNORECASE,
)

# ── Scoring weights ────────────────────────────────────

LONG_TEXT_THRESHOLD = 200  # words
MEDIUM_TEXT_THRESHOLD = 50
LONG_TEXT_SCORE = 3
MEDIUM_TEXT_SCORE = 1
CODE_BLOCK_SCORE = 2
CODE_KEYWORD_SCORE = 2
MULTI_STEP_SCORE = 1  # per match
ANALYSIS_SCORE = 2    # per match
MATH_SCORE = 2        # per match

# ── Tier thresholds ────────────────────────────────────

SIMPLE_MAX_SCORE = 2
STANDARD_MAX_SCORE = 6
# Anything above STANDARD_MAX_SCORE → Complex

# ── Budget downgrade thresholds (USD) ──────────────────

COMPLEX_DOWNGRADE_THRESHOLD = 0.10
STANDARD_DOWNGRADE_THRESHOLD = 0.05

# ── Default model tiers ───────────────────────────────

DEFAULT_MODELS: dict[Tier, list[str]] = {
    Tier.SIMPLE: ["gemini-2.5-flash", "gpt-4.1-nano"],
    Tier.STANDARD: ["claude-sonnet-4.6", "gpt-4.1-mini"],
    Tier.COMPLEX: ["claude-opus-4.6", "o4-mini"],
}


@dataclass
class RoutingDecision:
    """The result of a routing decision.

    Attributes:
        tier: Complexity tier the input was classified into.
        model: Primary model selected for this call.
        reason: Human-readable explanation of the routing decision.
        fallback: Backup model if the primary fails.
    """

    tier: Tier
    model: str
    reason: str
    fallback: str | None = None


@dataclass
class Router:
    """Classifies input complexity and selects the cheapest sufficient model.

    Uses pattern-based signals (word count, code blocks, analysis keywords,
    math keywords, multi-step phrases) to score complexity. No LLM calls,
    no external dependencies, sub-millisecond latency.
    """

    models: dict[Tier, list[str]] = field(default_factory=lambda: dict(DEFAULT_MODELS))

    def route(self, text: str, remaining_budget: float | None = None) -> RoutingDecision:
        """Classify input and return a routing decision.

        Args:
            text: The user input or latest message to classify.
            remaining_budget: USD remaining in the run budget. If low,
                the router downgrades to a cheaper tier automatically.

        Returns:
            A RoutingDecision with the selected model and tier.
        """
        tier = self._classify(text)
        tier = self._apply_budget_pressure(tier, remaining_budget)

        candidates = self.models.get(tier, self.models[Tier.STANDARD])
        return RoutingDecision(
            tier=tier,
            model=candidates[0],
            reason=f"classified as {tier.value}",
            fallback=candidates[1] if len(candidates) > 1 else None,
        )

    def _classify(self, text: str) -> Tier:
        """Score the input text and return a complexity tier."""
        score = 0
        lower = text.lower()
        words = lower.split()

        # Length signals
        if len(words) > LONG_TEXT_THRESHOLD:
            score += LONG_TEXT_SCORE
        elif len(words) > MEDIUM_TEXT_THRESHOLD:
            score += MEDIUM_TEXT_SCORE

        # Code signals
        if "```" in text:
            score += CODE_BLOCK_SCORE
        if any(keyword in text for keyword in ("def ", "function ", "class ")):
            score += CODE_KEYWORD_SCORE

        # Keyword signals (word-boundary matching)
        for phrase in MULTI_STEP_PHRASES:
            if phrase in lower:
                score += MULTI_STEP_SCORE
        score += len(_ANALYSIS_RE.findall(lower)) * ANALYSIS_SCORE
        score += len(_MATH_RE.findall(lower)) * MATH_SCORE

        if score <= SIMPLE_MAX_SCORE:
            return Tier.SIMPLE
        if score <= STANDARD_MAX_SCORE:
            return Tier.STANDARD
        return Tier.COMPLEX

    @staticmethod
    def _apply_budget_pressure(tier: Tier, remaining_budget: float | None) -> Tier:
        """Downgrade tier when budget is running low."""
        if remaining_budget is None:
            return tier
        if tier == Tier.COMPLEX and remaining_budget < COMPLEX_DOWNGRADE_THRESHOLD:
            tier = Tier.STANDARD
        if tier == Tier.STANDARD and remaining_budget < STANDARD_DOWNGRADE_THRESHOLD:
            tier = Tier.SIMPLE
        return tier
