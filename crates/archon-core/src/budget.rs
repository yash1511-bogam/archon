use crate::types::BudgetConfig;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BudgetError {
    #[error("budget exceeded: spent ${spent:.4} of ${limit:.4} limit")]
    Exceeded { spent: f64, limit: f64 },
    #[error("budget warning: spent ${spent:.4} ({pct:.0}% of ${limit:.4})")]
    Warning { spent: f64, limit: f64, pct: f64 },
}

/// Tracks spending and enforces budget limits.
#[derive(Debug, Clone)]
pub struct BudgetTracker {
    config: BudgetConfig,
    run_spent: f64,
    day_spent: f64,
    month_spent: f64,
}

impl BudgetTracker {
    pub fn new(config: BudgetConfig) -> Self {
        Self {
            config,
            run_spent: 0.0,
            day_spent: 0.0,
            month_spent: 0.0,
        }
    }

    /// Check if a proposed cost would exceed any budget limit.
    pub fn check(&self, proposed_cost: f64) -> Result<(), BudgetError> {
        if let Some(limit) = self.config.max_per_run {
            let after = self.run_spent + proposed_cost;
            if after > limit {
                return Err(BudgetError::Exceeded { spent: self.run_spent, limit });
            }
            let pct = after / limit;
            if pct >= self.config.warn_at_fraction {
                tracing::warn!(spent = self.run_spent, limit, pct, "budget warning");
            }
        }
        if let Some(limit) = self.config.max_per_day {
            if self.day_spent + proposed_cost > limit {
                return Err(BudgetError::Exceeded { spent: self.day_spent, limit });
            }
        }
        if let Some(limit) = self.config.max_per_month {
            if self.month_spent + proposed_cost > limit {
                return Err(BudgetError::Exceeded { spent: self.month_spent, limit });
            }
        }
        Ok(())
    }

    /// Record actual spending after a successful LLM call.
    pub fn record(&mut self, cost: f64) {
        self.run_spent += cost;
        self.day_spent += cost;
        self.month_spent += cost;
    }

    pub fn run_spent(&self) -> f64 {
        self.run_spent
    }

    pub fn remaining_run_budget(&self) -> Option<f64> {
        self.config.max_per_run.map(|limit| (limit - self.run_spent).max(0.0))
    }
}
