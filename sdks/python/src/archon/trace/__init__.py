"""Built-in trace store — SQLite-backed, zero external dependencies.

Records every agent step, audit event, and run summary.
Configurable retention. Query by run_id, time range, cost.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from archon.types import Step


@dataclass
class RunSummary:
    run_id: str
    agent: str
    total_steps: int
    total_cost: float
    total_tokens: int
    total_latency_ms: int
    started_at: str
    finished_at: str | None


@dataclass
class AuditEntry:
    run_id: str
    agent: str
    action: str
    detail: str | None
    timestamp: str


_SCHEMA = """\
CREATE TABLE IF NOT EXISTS steps (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    agent TEXT NOT NULL,
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
CREATE INDEX IF NOT EXISTS idx_steps_run ON steps(run_id);
CREATE INDEX IF NOT EXISTS idx_steps_ts ON steps(timestamp);
CREATE INDEX IF NOT EXISTS idx_steps_agent ON steps(agent);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    prompt TEXT,
    output TEXT,
    total_cost REAL NOT NULL DEFAULT 0,
    total_steps INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    total_latency_ms INTEGER NOT NULL DEFAULT 0,
    started_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_ts ON runs(started_at);
CREATE INDEX IF NOT EXISTS idx_runs_agent ON runs(agent);

CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    agent TEXT NOT NULL,
    action TEXT NOT NULL,
    detail TEXT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_audit_run ON audit(run_id);
"""


class TraceStore:
    """SQLite-backed trace store. Zero external dependencies."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is None:
            self._conn = sqlite3.connect(":memory:")
        else:
            p = Path(db_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(p))

        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def start_run(self, run_id: str, agent: str, prompt: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR IGNORE INTO runs (run_id, agent, prompt, started_at) VALUES (?, ?, ?, ?)",
            (run_id, agent, prompt, now),
        )
        self._conn.commit()

    def finish_run(
        self, run_id: str, output: str, total_cost: float,
        total_steps: int, total_tokens: int, total_latency_ms: int,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """UPDATE runs SET output=?, total_cost=?, total_steps=?, total_tokens=?,
               total_latency_ms=?, finished_at=? WHERE run_id=?""",
            (output, total_cost, total_steps, total_tokens, total_latency_ms, now, run_id),
        )
        self._conn.commit()

    def record_step(self, agent: str, run_id: str, step: Step) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO steps
               (id, run_id, agent, model, tier, input_tokens, output_tokens,
                cost_usd, latency_ms, tool_call, cached, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (step.id, run_id, agent, step.model, step.tier.value,
             step.input_tokens, step.output_tokens, step.cost_usd,
             step.latency_ms, step.tool_call, int(step.cached),
             step.timestamp.isoformat()),
        )
        self._conn.commit()

    def audit(self, run_id: str, agent: str, action: str, detail: str | None = None) -> None:
        self._conn.execute(
            "INSERT INTO audit (run_id, agent, action, detail) VALUES (?, ?, ?, ?)",
            (run_id, agent, action, detail),
        )
        self._conn.commit()

    # ── Query API ──────────────────────────────────────

    def get_run(self, run_id: str) -> RunSummary | None:
        row = self._conn.execute(
            "SELECT run_id, agent, total_steps, total_cost, total_tokens, total_latency_ms, started_at, finished_at FROM runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if not row:
            return None
        return RunSummary(*row)

    def list_runs(self, *, limit: int = 50, agent: str | None = None) -> list[RunSummary]:
        if agent:
            rows = self._conn.execute(
                "SELECT run_id, agent, total_steps, total_cost, total_tokens, total_latency_ms, started_at, finished_at FROM runs WHERE agent=? ORDER BY started_at DESC LIMIT ?",
                (agent, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT run_id, agent, total_steps, total_cost, total_tokens, total_latency_ms, started_at, finished_at FROM runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [RunSummary(*r) for r in rows]

    def get_steps(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, model, tier, input_tokens, output_tokens, cost_usd, latency_ms, tool_call, cached, timestamp FROM steps WHERE run_id=? ORDER BY timestamp",
            (run_id,),
        ).fetchall()
        return [
            {"id": r[0], "model": r[1], "tier": r[2], "input_tokens": r[3],
             "output_tokens": r[4], "cost_usd": r[5], "latency_ms": r[6],
             "tool_call": r[7], "cached": bool(r[8]), "timestamp": r[9]}
            for r in rows
        ]

    def get_audit(self, run_id: str) -> list[AuditEntry]:
        rows = self._conn.execute(
            "SELECT run_id, agent, action, detail, timestamp FROM audit WHERE run_id=? ORDER BY timestamp",
            (run_id,),
        ).fetchall()
        return [AuditEntry(*r) for r in rows]

    def stats(self) -> dict[str, Any]:
        """Aggregate statistics across all runs."""
        row = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(total_cost),0), COALESCE(SUM(total_tokens),0), COALESCE(SUM(total_steps),0) FROM runs"
        ).fetchone()
        top_models = self._conn.execute(
            "SELECT model, COUNT(*), SUM(cost_usd) FROM steps GROUP BY model ORDER BY COUNT(*) DESC LIMIT 10"
        ).fetchall()
        return {
            "total_runs": row[0],
            "total_cost_usd": round(row[1], 6),
            "total_tokens": row[2],
            "total_steps": row[3],
            "top_models": [{"model": m[0], "calls": m[1], "cost": round(m[2], 6)} for m in top_models],
        }

    def purge_before(self, before: datetime) -> int:
        """Delete runs and steps older than the given datetime. Returns count deleted."""
        ts = before.isoformat()
        count = self._conn.execute("SELECT COUNT(*) FROM runs WHERE started_at < ?", (ts,)).fetchone()[0]
        self._conn.execute("DELETE FROM steps WHERE run_id IN (SELECT run_id FROM runs WHERE started_at < ?)", (ts,))
        self._conn.execute("DELETE FROM audit WHERE run_id IN (SELECT run_id FROM runs WHERE started_at < ?)", (ts,))
        self._conn.execute("DELETE FROM runs WHERE started_at < ?", (ts,))
        self._conn.commit()
        return count

    def close(self) -> None:
        self._conn.close()
