"""Budget enforcement for agent runs.

Budgets are hard limits, not suggestions. When an agent approaches
its budget, the router downgrades to cheaper models. When the budget
is exceeded, the agent stops and returns a partial result.

All three limits (per-run, per-day, per-month) are enforced.
Day/month tracking resets automatically on calendar boundaries.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from pydantic import BaseModel, PrivateAttr

# Default warning threshold — alert at 80% of any budget limit
DEFAULT_WARN_THRESHOLD = 0.8


class BudgetExceeded(Exception):
    """Raised when a proposed LLM call would exceed the budget."""

    def __init__(self, spent: float, limit: float, scope: str = "run") -> None:
        self.spent = spent
        self.limit = limit
        self.scope = scope
        super().__init__(
            f"Budget exceeded ({scope}): spent ${spent:.4f} of ${limit:.4f} limit"
        )


class Budget(BaseModel):
    """Budget configuration with hard enforcement.

    All three limits are enforced independently. Day and month
    tracking resets automatically on calendar boundaries.

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
    _run_spent: float = PrivateAttr(default=0.0)
    _day_spent: float = PrivateAttr(default=0.0)
    _month_spent: float = PrivateAttr(default=0.0)
    _current_day: int = PrivateAttr(default=0)
    _current_month: int = PrivateAttr(default=0)

    def model_post_init(self, __context: object) -> None:
        """Initialize day/month tracking to current calendar period."""
        now = datetime.now(timezone.utc)
        self._current_day = now.timetuple().tm_yday  # day-of-year (1-366)
        self._current_month = now.month

    def check(self, proposed_cost: float) -> None:
        """Verify that a proposed cost won't exceed any budget limit.

        Checks all three limits: per-run, per-day, per-month.
        Day/month counters reset automatically on calendar boundaries.

        Raises:
            BudgetExceeded: If spending the proposed amount would exceed any limit.
        """
        self._reset_if_new_period()

        if self.max_per_run is not None:
            if self._run_spent + proposed_cost > self.max_per_run:
                raise BudgetExceeded(self._run_spent, self.max_per_run, "run")

        if self.max_per_day is not None:
            if self._day_spent + proposed_cost > self.max_per_day:
                raise BudgetExceeded(self._day_spent, self.max_per_day, "day")

        if self.max_per_month is not None:
            if self._month_spent + proposed_cost > self.max_per_month:
                raise BudgetExceeded(self._month_spent, self.max_per_month, "month")

    def record(self, cost: float) -> None:
        """Record actual spending after a successful LLM call."""
        self._reset_if_new_period()
        self._run_spent += cost
        self._day_spent += cost
        self._month_spent += cost

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

    def _reset_if_new_period(self) -> None:
        """Reset day/month counters if the calendar period has changed."""
        now = datetime.now(timezone.utc)
        day_of_year = now.timetuple().tm_yday

        if day_of_year != self._current_day:
            self._day_spent = 0.0
            self._current_day = day_of_year

        if now.month != self._current_month:
            self._month_spent = 0.0
            self._current_month = now.month
