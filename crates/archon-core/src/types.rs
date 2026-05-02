use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Complexity tier for model routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    Simple,
    Standard,
    Complex,
}

/// A single step in an agent execution trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    pub id: String,
    pub model: String,
    pub tier: Tier,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost_usd: f64,
    pub latency_ms: u64,
    pub tool_call: Option<String>,
    pub cached: bool,
    pub timestamp: DateTime<Utc>,
}

/// Result of a complete agent run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    pub run_id: String,
    pub output: String,
    pub steps: Vec<Step>,
    pub total_cost_usd: f64,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_latency_ms: u64,
    pub model_usage: std::collections::HashMap<String, u64>,
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
}

impl RunResult {
    pub fn new() -> Self {
        let now = Utc::now();
        Self {
            run_id: Uuid::new_v4().to_string(),
            output: String::new(),
            steps: Vec::new(),
            total_cost_usd: 0.0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            total_latency_ms: 0,
            model_usage: std::collections::HashMap::new(),
            started_at: now,
            finished_at: now,
        }
    }

    pub fn add_step(&mut self, step: Step) {
        self.total_cost_usd += step.cost_usd;
        self.total_input_tokens += step.input_tokens;
        self.total_output_tokens += step.output_tokens;
        self.total_latency_ms += step.latency_ms;
        *self.model_usage.entry(step.model.clone()).or_insert(0) += 1;
        self.steps.push(step);
    }

    pub fn finish(&mut self, output: String) {
        self.output = output;
        self.finished_at = Utc::now();
    }
}

/// Budget configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetConfig {
    pub max_per_run: Option<f64>,
    pub max_per_day: Option<f64>,
    pub max_per_month: Option<f64>,
    pub warn_at_fraction: f64,
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            max_per_run: None,
            max_per_day: None,
            max_per_month: None,
            warn_at_fraction: 0.8,
        }
    }
}
