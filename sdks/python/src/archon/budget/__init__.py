"""Budget enforcement for agent runs.

Budgets are hard limits, not suggestions. When an agent approaches
its budget, the router downgrades to cheaper models. When the budget
is exceeded, the agent stops and returns a partial result.
"""

from __future__ import annotations

from pydantic import BaseModel

# Default warning threshold — alert at 80% of any budget limit
DEFAULT_WARN_THRESHOLD = 0.8


class BudgetExceeded(Exception):
    """Raised when a proposed LLM call would exceed the budget."""

    def __init__(self, spent: float, limit: float) -> None:
        self.spent = spent
        self.limit = limit
        super().__init__(f"Budget exceeded: spent ${spent:.4f} of ${limit:.4f} limit")


class Budget(BaseModel):
    """Budget configuration with hard enforcement.

    Attributes:
        max_per_run: Maximum USD spend for a single agent.run() call.
        max_per_day: Maximum USD spend per calendar day.
        max_per_month: Maximum USD spend per calendar month.
        warn_at: Fraction (0.0–1.0) at which to emit a warning.
    """

    max_per_run: float | None = None
    max_per_day: float | None = None
    max_per_month: float | None = None
    warn_at: float = DEFAULT_WARN_THRESHOLD

    # Private mutable state — not serialized by Pydantic
    _run_spent: float = 0.0

    def check(self, proposed_cost: float) -> None:
        """Verify that a proposed cost won't exceed the run budget.

        Raises:
            BudgetExceeded: If spending the proposed amount would exceed max_per_run.
        """
        if self.max_per_run is not None:
            if self._run_spent + proposed_cost > self.max_per_run:
                raise BudgetExceeded(self._run_spent, self.max_per_run)

    def record(self, cost: float) -> None:
        """Record actual spending after a successful LLM call."""
        self._run_spent += cost

    @property
    def spent(self) -> float:
        """Total USD spent in the current run."""
        return self._run_spent

    @property
    def remaining(self) -> float | None:
        """USD remaining in the run budget, or None if no limit is set."""
        if self.max_per_run is None:
            return None
        return max(0.0, self.max_per_run - self._run_spent)
