"""Budget enforcement for agent runs."""

from __future__ import annotations

from pydantic import BaseModel


class BudgetExceeded(Exception):
    """Raised when an agent exceeds its budget."""

    def __init__(self, spent: float, limit: float) -> None:
        self.spent = spent
        self.limit = limit
        super().__init__(f"Budget exceeded: spent ${spent:.4f} of ${limit:.4f} limit")


class Budget(BaseModel):
    """Budget configuration with hard enforcement."""

    max_per_run: float | None = None
    max_per_day: float | None = None
    max_per_month: float | None = None
    warn_at: float = 0.8

    _run_spent: float = 0.0

    def check(self, proposed_cost: float) -> None:
        """Raise BudgetExceeded if proposed cost would exceed limits."""
        if self.max_per_run is not None:
            if self._run_spent + proposed_cost > self.max_per_run:
                raise BudgetExceeded(self._run_spent, self.max_per_run)

    def record(self, cost: float) -> None:
        """Record actual spending."""
        self._run_spent += cost

    @property
    def spent(self) -> float:
        return self._run_spent

    @property
    def remaining(self) -> float | None:
        if self.max_per_run is None:
            return None
        return max(0.0, self.max_per_run - self._run_spent)
