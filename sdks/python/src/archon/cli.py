"""archon CLI — production tooling for AI agents.

Commands::

    archon traces list          # List recent agent runs
    archon traces show <id>     # Show details of a specific run
    archon traces stats         # Aggregate statistics
    archon traces purge         # Delete old traces
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import click

from archon.trace import TraceStore

DEFAULT_TRACE_DB = Path.home() / ".archon" / "traces.db"

# ── Table formatting constants ─────────────────────────

LIST_HEADER = f"{'RUN ID':<38} {'AGENT':<15} {'STEPS':>5} {'COST':>10} {'TOKENS':>8} {'LATENCY':>10} {'STARTED'}"
LIST_SEPARATOR = "─" * 110
STEP_HEADER = f"{'#':>3} {'MODEL':<25} {'TIER':<10} {'TOKENS':>8} {'COST':>10} {'LATENCY':>10} {'TOOL'}"
STEP_SEPARATOR = "─" * 90


def _open_store(db: str | None) -> TraceStore:
    """Open the trace database, or exit with a helpful message if it doesn't exist."""
    path = Path(db) if db else DEFAULT_TRACE_DB
    if not path.exists():
        click.echo(f"No trace database found at {path}", err=True)
        click.echo("Run an agent with a TraceStore first, or specify --db.", err=True)
        sys.exit(1)
    return TraceStore(path)


# ── CLI groups ─────────────────────────────────────────

@click.group()
@click.version_option(package_name="archon-ai")
def main() -> None:
    """Archon — the production harness for AI agents."""


@main.group()
def traces() -> None:
    """Query and manage agent execution traces."""


@main.command("dashboard")
@click.option("--port", "-p", default=8080, help="Port to serve on.")
@click.option("--db", default=None, help="Path to trace database.")
def dashboard_cmd(port: int, db: str | None) -> None:
    """Start the built-in web dashboard."""
    from archon.dashboard import start_dashboard
    start_dashboard(port=port, db_path=db)


# ── archon traces list ─────────────────────────────────

@traces.command("list")
@click.option("--db", default=None, help="Path to trace database.")
@click.option("--limit", "-n", default=20, help="Number of runs to show.")
@click.option("--agent", "-a", default=None, help="Filter by agent name.")
@click.option("--json-output", "as_json", is_flag=True, help="Output as JSON.")
def traces_list(db: str | None, limit: int, agent: str | None, as_json: bool) -> None:
    """List recent agent runs."""
    store = _open_store(db)
    runs = store.list_runs(limit=limit, agent=agent)
    store.close()

    if as_json:
        data = [
            {
                "run_id": r.run_id, "agent": r.agent, "steps": r.total_steps,
                "cost": r.total_cost, "tokens": r.total_tokens,
                "latency_ms": r.total_latency_ms, "started_at": r.started_at,
            }
            for r in runs
        ]
        click.echo(json.dumps(data, indent=2))
        return

    if not runs:
        click.echo("No runs found.")
        return

    click.echo(LIST_HEADER)
    click.echo(LIST_SEPARATOR)
    for run in runs:
        started = run.started_at[:19] if run.started_at else "—"
        click.echo(
            f"{run.run_id:<38} {run.agent:<15} {run.total_steps:>5}"
            f" ${run.total_cost:>9.4f} {run.total_tokens:>8}"
            f" {run.total_latency_ms:>9}ms {started}"
        )


# ── archon traces show ─────────────────────────────────

@traces.command("show")
@click.argument("run_id")
@click.option("--db", default=None, help="Path to trace database.")
@click.option("--json-output", "as_json", is_flag=True, help="Output as JSON.")
def traces_show(run_id: str, db: str | None, as_json: bool) -> None:
    """Show details of a specific run."""
    store = _open_store(db)
    run = store.get_run(run_id)
    if not run:
        click.echo(f"Run {run_id} not found.", err=True)
        sys.exit(1)

    steps = store.get_steps(run_id)
    audit_entries = store.get_audit(run_id)
    store.close()

    if as_json:
        click.echo(json.dumps({
            "run": {
                "run_id": run.run_id, "agent": run.agent,
                "steps": run.total_steps, "cost": run.total_cost,
                "tokens": run.total_tokens, "latency_ms": run.total_latency_ms,
                "started_at": run.started_at, "finished_at": run.finished_at,
            },
            "steps": steps,
            "audit": [
                {"action": a.action, "detail": a.detail, "timestamp": a.timestamp}
                for a in audit_entries
            ],
        }, indent=2))
        return

    # Run summary
    click.echo(f"Run:      {run.run_id}")
    click.echo(f"Agent:    {run.agent}")
    click.echo(f"Cost:     ${run.total_cost:.4f}")
    click.echo(f"Steps:    {run.total_steps}")
    click.echo(f"Tokens:   {run.total_tokens:,}")
    click.echo(f"Latency:  {run.total_latency_ms}ms")
    click.echo(f"Started:  {run.started_at}")
    click.echo(f"Finished: {run.finished_at or '—'}")

    # Step-by-step breakdown
    if steps:
        click.echo(f"\n{STEP_HEADER}")
        click.echo(STEP_SEPARATOR)
        for i, step in enumerate(steps):
            total_tokens = step["input_tokens"] + step["output_tokens"]
            tool_name = step["tool_call"] or "—"
            cache_marker = " ⚡" if step["cached"] else ""
            click.echo(
                f"{i + 1:>3} {step['model']:<25} {step['tier']:<10}"
                f" {total_tokens:>8} ${step['cost_usd']:>9.4f}"
                f" {step['latency_ms']:>9}ms {tool_name}{cache_marker}"
            )

    # Audit log
    if audit_entries:
        click.echo("\nAudit Log:")
        for entry in audit_entries:
            detail_str = f" — {entry.detail}" if entry.detail else ""
            click.echo(f"  [{entry.timestamp[:19]}] {entry.action}{detail_str}")


# ── archon traces stats ────────────────────────────────

@traces.command("stats")
@click.option("--db", default=None, help="Path to trace database.")
@click.option("--json-output", "as_json", is_flag=True, help="Output as JSON.")
def traces_stats(db: str | None, as_json: bool) -> None:
    """Show aggregate statistics across all runs."""
    store = _open_store(db)
    statistics = store.stats()
    store.close()

    if as_json:
        click.echo(json.dumps(statistics, indent=2))
        return

    click.echo(f"Total runs:   {statistics['total_runs']}")
    click.echo(f"Total cost:   ${statistics['total_cost_usd']:.4f}")
    click.echo(f"Total tokens: {statistics['total_tokens']:,}")
    click.echo(f"Total steps:  {statistics['total_steps']}")

    if statistics["top_models"]:
        click.echo("\nTop Models:")
        for model in statistics["top_models"]:
            click.echo(f"  {model['model']:<30} {model['calls']:>6} calls  ${model['cost']:.4f}")


# ── archon traces purge ────────────────────────────────

@traces.command("purge")
@click.option("--db", default=None, help="Path to trace database.")
@click.option("--before-days", type=int, required=True, help="Delete runs older than N days.")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
def traces_purge(db: str | None, before_days: int, yes: bool) -> None:
    """Delete traces older than a specified number of days."""
    store = _open_store(db)
    cutoff = datetime.now(timezone.utc) - timedelta(days=before_days)

    if not yes:
        click.confirm(f"Delete all traces before {cutoff.date()}?", abort=True)

    deleted_count = store.purge_before(cutoff)
    store.close()
    click.echo(f"Purged {deleted_count} runs.")
