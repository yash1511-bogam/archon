use crate::types::Tier;
use serde::{Deserialize, Serialize};

/// Routing decision with reasoning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub tier: Tier,
    pub model: String,
    pub reason: String,
    pub fallback: Option<String>,
}

/// Configuration for model tiers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    pub simple: Vec<String>,
    pub standard: Vec<String>,
    pub complex: Vec<String>,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            simple: vec!["gemini-2.5-flash".into(), "gpt-4.1-nano".into()],
            standard: vec!["claude-sonnet-4.6".into(), "gpt-4.1-mini".into()],
            complex: vec!["claude-opus-4.6".into(), "o4-mini".into()],
        }
    }
}

/// Pattern-based complexity classifier. Zero LLM calls, zero latency.
pub struct Router {
    config: RouterConfig,
}

impl Router {
    pub fn new(config: RouterConfig) -> Self {
        Self { config }
    }

    /// Classify input complexity and return a routing decision.
    pub fn route(&self, input: &str, remaining_budget: Option<f64>) -> RoutingDecision {
        let tier = self.classify(input);

        // Downgrade if budget pressure
        let effective_tier = match (tier, remaining_budget) {
            (Tier::Complex, Some(b)) if b < 0.10 => Tier::Standard,
            (Tier::Standard, Some(b)) if b < 0.05 => Tier::Simple,
            _ => tier,
        };

        let models = match effective_tier {
            Tier::Simple => &self.config.simple,
            Tier::Standard => &self.config.standard,
            Tier::Complex => &self.config.complex,
        };

        let model = models.first().cloned().unwrap_or_else(|| "gpt-4.1-mini".into());
        let fallback = models.get(1).cloned();

        RoutingDecision {
            tier: effective_tier,
            model,
            reason: format!("classified as {effective_tier:?}"),
            fallback,
        }
    }

    /// Pattern-based complexity classification using signal counting.
    fn classify(&self, input: &str) -> Tier {
        let mut score: u32 = 0;
        let words: Vec<&str> = input.split_whitespace().collect();
        let len = words.len();

        // Length signals
        if len > 200 { score += 3; }
        else if len > 50 { score += 1; }

        // Code signals
        if input.contains("```") { score += 2; }
        if input.contains("def ") || input.contains("function ") || input.contains("class ") {
            score += 2;
        }

        // Multi-step signals
        let multi_step = ["step 1", "first,", "then,", "finally,", "after that", "next,"];
        for kw in &multi_step {
            if input.to_lowercase().contains(kw) { score += 1; }
        }

        // Analysis signals
        let analysis = ["analyze", "compare", "evaluate", "architect", "design", "optimize", "debug", "refactor"];
        for kw in &analysis {
            if input.to_lowercase().contains(kw) { score += 2; }
        }

        // Math/logic signals
        let math = ["calculate", "prove", "derive", "equation", "algorithm", "complexity"];
        for kw in &math {
            if input.to_lowercase().contains(kw) { score += 2; }
        }

        match score {
            0..=2 => Tier::Simple,
            3..=6 => Tier::Standard,
            _ => Tier::Complex,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_query() {
        let router = Router::new(RouterConfig::default());
        let decision = router.route("What is 2+2?", None);
        assert_eq!(decision.tier, Tier::Simple);
    }

    #[test]
    fn complex_query() {
        let router = Router::new(RouterConfig::default());
        let decision = router.route(
            "Analyze and compare the architecture of microservices vs monolith, \
             then design an optimal migration strategy with code examples",
            None,
        );
        assert_eq!(decision.tier, Tier::Complex);
    }

    #[test]
    fn budget_pressure_downgrades() {
        let router = Router::new(RouterConfig::default());
        // With very low budget, complex gets downgraded
        let decision = router.route(
            "Analyze this complex architecture and optimize it",
            Some(0.03),
        );
        // Complex → Standard (budget < 0.10), then Standard → Simple (budget < 0.05)
        assert_eq!(decision.tier, Tier::Simple);
    }
}
