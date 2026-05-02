use crate::types::Step;
use rusqlite::{params, Connection};
use std::path::Path;

/// Schema SQL shared between file-backed and in-memory stores.
const SCHEMA: &str = "\
CREATE TABLE IF NOT EXISTS traces (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    model TEXT NOT NULL,
    tier TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    latency_ms INTEGER NOT NULL,
    tool_call TEXT,
    cached INTEGER NOT NULL DEFAULT 0,
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traces_run ON traces(run_id);
CREATE INDEX IF NOT EXISTS idx_traces_ts ON traces(timestamp);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    agent TEXT NOT NULL,
    action TEXT NOT NULL,
    detail TEXT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now'))
);";

/// SQLite-backed trace store. Zero external dependencies.
pub struct TraceStore {
    conn: Connection,
}

impl TraceStore {
    pub fn new(path: &Path) -> Result<Self, rusqlite::Error> {
        let conn = Connection::open(path)?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self { conn })
    }

    /// Open an in-memory store (for testing).
    pub fn in_memory() -> Result<Self, rusqlite::Error> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self { conn })
    }

    pub fn record_step(&self, run_id: &str, step: &Step) -> Result<(), rusqlite::Error> {
        self.conn.execute(
            "INSERT INTO traces (id, run_id, model, tier, input_tokens, output_tokens, cost_usd, latency_ms, tool_call, cached, timestamp)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                step.id,
                run_id,
                step.model,
                serde_json::to_string(&step.tier).unwrap_or_default(),
                step.input_tokens as i64,
                step.output_tokens as i64,
                step.cost_usd,
                step.latency_ms as i64,
                step.tool_call,
                step.cached,
                step.timestamp.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    pub fn audit(&self, run_id: &str, agent: &str, action: &str, detail: Option<&str>) -> Result<(), rusqlite::Error> {
        self.conn.execute(
            "INSERT INTO audit_log (run_id, agent, action, detail) VALUES (?1, ?2, ?3, ?4)",
            params![run_id, agent, action, detail],
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Tier;
    use chrono::Utc;

    #[test]
    fn record_and_audit() {
        let store = TraceStore::in_memory().unwrap();
        let step = Step {
            id: "step-1".into(),
            model: "gemini-flash".into(),
            tier: Tier::Simple,
            input_tokens: 100,
            output_tokens: 50,
            cost_usd: 0.001,
            latency_ms: 200,
            tool_call: None,
            cached: false,
            timestamp: Utc::now(),
        };
        store.record_step("run-1", &step).unwrap();
        store.audit("run-1", "researcher", "tool_call", Some("search_web")).unwrap();
    }
}
