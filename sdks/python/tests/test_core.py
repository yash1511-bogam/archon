"""Tests for Archon core modules (no LLM calls)."""

from archon import Budget, Router, Tier, tool
from archon.budget import BudgetExceeded
from archon.types import AgentResult, Step

import pytest


def test_tool_decorator():
    @tool
    def search(query: str) -> str:
        """Search the web."""
        return f"results for {query}"

    assert search.name == "search"
    assert search.description == "Search the web."
    assert "query" in search.parameters
    schema = search.to_schema()
    assert schema["function"]["name"] == "search"


def test_router_simple():
    r = Router()
    decision = r.route("Hello!")
    assert decision.tier == Tier.SIMPLE


def test_router_complex():
    r = Router()
    decision = r.route(
        "Analyze and compare the architecture, then design an optimized solution with code"
    )
    assert decision.tier == Tier.COMPLEX


def test_router_budget_downgrade():
    r = Router()
    decision = r.route("Analyze this architecture", remaining_budget=0.03)
    assert decision.tier == Tier.SIMPLE  # Standard → Simple due to budget < 0.05


def test_budget_enforcement():
    b = Budget(max_per_run=0.10)
    b.check(0.05)  # OK
    b.record(0.05)
    b.check(0.04)  # OK
    b.record(0.04)
    with pytest.raises(BudgetExceeded):
        b.check(0.02)  # Exceeds 0.10


def test_budget_remaining():
    b = Budget(max_per_run=1.00)
    assert b.remaining == 1.00
    b.record(0.30)
    assert b.remaining == pytest.approx(0.70)


def test_budget_no_limit():
    b = Budget()
    b.check(1000.0)  # No limit, always OK
    assert b.remaining is None


def test_agent_result():
    result = AgentResult(run_id="test-1", output="hello")
    assert result.cost == 0.0
    assert result.step_count == 0
    assert "test-1" in result.trace_url


def test_step_model():
    step = Step(id="s1", model="gemini-flash", tier=Tier.SIMPLE)
    assert step.cost_usd == 0.0
    assert step.cached is False
