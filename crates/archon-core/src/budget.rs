use crate::types::BudgetConfig;
use chrono::{Datelike, Utc};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BudgetError {
    #[error("budget exceeded ({scope}): spent ${spent:.4} of ${limit:.4} limit")]
    Exceeded {
        spent: f64,
        limit: f64,
        scope: String,
    },
    #[error("budget warning: spent ${spent:.4} ({pct:.0}% of ${limit:.4})")]
    Warning { spent: f64, limit: f64, pct: f64 },
}

/// Tracks spending and enforces budget limits.
///
/// Day and month counters reset automatically on calendar boundaries,
/// matching the Python and TypeScript implementations.
#[derive(Debug, Clone)]
pub struct BudgetTracker {
    config: BudgetConfig,
    run_spent: f64,
    day_spent: f64,
    month_spent: f64,
    current_day: u32,   // day-of-year (1–366)
    current_month: u32, // month (1–12)
}

impl BudgetTracker {
    pub fn new(config: BudgetConfig) -> Self {
        let now = Utc::now();
        Self {
            config,
            run_spent: 0.0,
            day_spent: 0.0,
            month_spent: 0.0,
            current_day: now.ordinal(),
            current_month: now.month(),
        }
    }

    /// Check if a proposed cost would exceed any budget limit.
    pub fn check(&self, proposed_cost: f64) -> Result<(), BudgetError> {
        if let Some(limit) = self.config.max_per_run {
            let after = self.run_spent + proposed_cost;
            if after > limit {
                return Err(BudgetError::Exceeded {
                    spent: self.run_spent,
                    limit,
                    scope: "run".into(),
                });
            }
            let pct = after / limit;
            if pct >= self.config.warn_at_fraction {
                tracing::warn!(spent = self.run_spent, limit, pct, "budget warning");
            }
        }
        if let Some(limit) = self.config.max_per_day {
            if self.day_spent + proposed_cost > limit {
                return Err(BudgetError::Exceeded {
                    spent: self.day_spent,
                    limit,
                    scope: "day".into(),
                });
            }
        }
        if let Some(limit) = self.config.max_per_month {
            if self.month_spent + proposed_cost > limit {
                return Err(BudgetError::Exceeded {
                    spent: self.month_spent,
                    limit,
                    scope: "month".into(),
                });
            }
        }
        Ok(())
    }

    /// Record actual spending after a successful LLM call.
    pub fn record(&mut self, cost: f64) {
        self.reset_if_new_period();
        self.run_spent += cost;
        self.day_spent += cost;
        self.month_spent += cost;
    }

    pub fn run_spent(&self) -> f64 {
        self.run_spent
    }

    pub fn remaining_run_budget(&self) -> Option<f64> {
        self.config
            .max_per_run
            .map(|limit| (limit - self.run_spent).max(0.0))
    }

    /// Reset day/month counters when the calendar period changes.
    fn reset_if_new_period(&mut self) {
        let now = Utc::now();
        let day = now.ordinal();
        let month = now.month();

        if day != self.current_day {
            self.day_spent = 0.0;
            self.current_day = day;
        }
        if month != self.current_month {
            self.month_spent = 0.0;
            self.current_month = month;
        }
    }
}
