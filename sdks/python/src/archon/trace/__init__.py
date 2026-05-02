"""Built-in trace store — SQLite-backed, zero external dependencies.

Records every agent step, audit event, and run summary.
Configurable retention. Query by run_id, time range, agent name.

Every table uses WAL mode for concurrent read/write safety.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from archon.types import Step

# ── Schema ─────────────────────────────────────────────

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS steps (
    id          TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL,
    agent       TEXT NOT NULL,
    model       TEXT NOT NULL,
    tier        TEXT NOT NULL,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd    REAL NOT NULL,
    latency_ms  INTEGER NOT NULL,
    tool_call   TEXT,
    cached      INTEGER NOT NULL DEFAULT 0,
    timestamp   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_steps_run   ON steps(run_id);
CREATE INDEX IF NOT EXISTS idx_steps_ts    ON steps(timestamp);
CREATE INDEX IF NOT EXISTS idx_steps_agent ON steps(agent);

CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    agent       TEXT NOT NULL,
    prompt      TEXT,
    output      TEXT,
    total_cost       REAL    NOT NULL DEFAULT 0,
    total_steps      INTEGER NOT NULL DEFAULT 0,
    total_tokens     INTEGER NOT NULL DEFAULT 0,
    total_latency_ms INTEGER NOT NULL DEFAULT 0,
    started_at  TEXT NOT NULL,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_ts    ON runs(started_at);
CREATE INDEX IF NOT EXISTS idx_runs_agent ON runs(agent);

CREATE TABLE IF NOT EXISTS audit (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    TEXT NOT NULL,
    agent     TEXT NOT NULL,
    action    TEXT NOT NULL,
    detail    TEXT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_audit_run ON audit(run_id);
"""


# ── Data classes ───────────────────────────────────────

@dataclass
class RunSummary:
    """Summary of a single agent run, as stored in the database."""

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
    """A single entry in the immutable audit log."""

    run_id: str
    agent: str
    action: str
    detail: str | None
    timestamp: str


# ── Trace store ────────────────────────────────────────

class TraceStore:
    """SQLite-backed trace store with zero external dependencies.

    Stores execution traces, run summaries, and an immutable audit log.
    Uses WAL mode for safe concurrent access.

    Args:
        db_path: Path to SQLite file. None for in-memory (testing).
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._conn = self._open_db(db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ── Write operations ───────────────────────────────

    def start_run(self, run_id: str, agent: str, prompt: str) -> None:
        """Register a new agent run."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR IGNORE INTO runs (run_id, agent, prompt, started_at) VALUES (?, ?, ?, ?)",
            (run_id, agent, prompt, now),
        )
        self._conn.commit()

    def finish_run(
        self,
        run_id: str,
        output: str,
        total_cost: float,
        total_steps: int,
        total_tokens: int,
        total_latency_ms: int,
    ) -> None:
        """Mark a run as complete with final totals."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE runs SET output=?, total_cost=?, total_steps=?, total_tokens=?,"
            " total_latency_ms=?, finished_at=? WHERE run_id=?",
            (output, total_cost, total_steps, total_tokens, total_latency_ms, now, run_id),
        )
        self._conn.commit()

    def record_step(self, agent: str, run_id: str, step: Step) -> None:
        """Record a single execution step (LLM call, tool call, or cache hit)."""
        self._conn.execute(
            "INSERT OR REPLACE INTO steps"
            " (id, run_id, agent, model, tier, input_tokens, output_tokens,"
            "  cost_usd, latency_ms, tool_call, cached, timestamp)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (step.id, run_id, agent, step.model, step.tier.value,
             step.input_tokens, step.output_tokens, step.cost_usd,
             step.latency_ms, step.tool_call, int(step.cached),
             step.timestamp.isoformat()),
        )
        self._conn.commit()

    def audit(self, run_id: str, agent: str, action: str, detail: str | None = None) -> None:
        """Append an entry to the immutable audit log."""
        self._conn.execute(
            "INSERT INTO audit (run_id, agent, action, detail) VALUES (?, ?, ?, ?)",
            (run_id, agent, action, detail),
        )
        self._conn.commit()

    # ── Read operations ────────────────────────────────

    def get_run(self, run_id: str) -> RunSummary | None:
        """Fetch a single run summary by ID."""
        row = self._conn.execute(
            "SELECT run_id, agent, total_steps, total_cost, total_tokens,"
            " total_latency_ms, started_at, finished_at FROM runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        return RunSummary(*row) if row else None

    def list_runs(self, *, limit: int = 50, agent: str | None = None) -> list[RunSummary]:
        """List recent runs, optionally filtered by agent name."""
        query = (
            "SELECT run_id, agent, total_steps, total_cost, total_tokens,"
            " total_latency_ms, started_at, finished_at FROM runs"
        )
        params: list[Any] = []
        if agent:
            query += " WHERE agent=?"
            params.append(agent)
        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        return [RunSummary(*row) for row in self._conn.execute(query, params).fetchall()]

    def get_steps(self, run_id: str) -> list[dict[str, Any]]:
        """Fetch all steps for a run, ordered by timestamp."""
        rows = self._conn.execute(
            "SELECT id, model, tier, input_tokens, output_tokens, cost_usd,"
            " latency_ms, tool_call, cached, timestamp"
            " FROM steps WHERE run_id=? ORDER BY timestamp",
            (run_id,),
        ).fetchall()
        return [
            {
                "id": r[0], "model": r[1], "tier": r[2],
                "input_tokens": r[3], "output_tokens": r[4], "cost_usd": r[5],
                "latency_ms": r[6], "tool_call": r[7], "cached": bool(r[8]),
                "timestamp": r[9],
            }
            for r in rows
        ]

    def get_audit(self, run_id: str) -> list[AuditEntry]:
        """Fetch all audit entries for a run, ordered by timestamp."""
        rows = self._conn.execute(
            "SELECT run_id, agent, action, detail, timestamp"
            " FROM audit WHERE run_id=? ORDER BY timestamp",
            (run_id,),
        ).fetchall()
        return [AuditEntry(*r) for r in rows]

    def stats(self) -> dict[str, Any]:
        """Aggregate statistics across all runs."""
        totals = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(total_cost), 0),"
            " COALESCE(SUM(total_tokens), 0), COALESCE(SUM(total_steps), 0)"
            " FROM runs"
        ).fetchone()

        top_models = self._conn.execute(
            "SELECT model, COUNT(*), SUM(cost_usd)"
            " FROM steps GROUP BY model ORDER BY COUNT(*) DESC LIMIT 10"
        ).fetchall()

        return {
            "total_runs": totals[0],
            "total_cost_usd": round(totals[1], 6),
            "total_tokens": totals[2],
            "total_steps": totals[3],
            "top_models": [
                {"model": m[0], "calls": m[1], "cost": round(m[2], 6)}
                for m in top_models
            ],
        }

    # ── Maintenance ────────────────────────────────────

    def purge_before(self, before: datetime) -> int:
        """Delete runs, steps, and audit entries older than the given datetime.

        Returns:
            Number of runs deleted.
        """
        cutoff = before.isoformat()
        count = self._conn.execute(
            "SELECT COUNT(*) FROM runs WHERE started_at < ?", (cutoff,)
        ).fetchone()[0]

        old_runs_subquery = "SELECT run_id FROM runs WHERE started_at < ?"
        self._conn.execute(f"DELETE FROM steps WHERE run_id IN ({old_runs_subquery})", (cutoff,))
        self._conn.execute(f"DELETE FROM audit WHERE run_id IN ({old_runs_subquery})", (cutoff,))
        self._conn.execute("DELETE FROM runs WHERE started_at < ?", (cutoff,))
        self._conn.commit()
        return count

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    # ── Private helpers ────────────────────────────────

    @staticmethod
    def _open_db(db_path: Path | str | None) -> sqlite3.Connection:
        if db_path is None:
            return sqlite3.connect(":memory:")
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn
