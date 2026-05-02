"""archon CLI — production tooling for AI agents."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from archon.trace import TraceStore

DEFAULT_DB = Path.home() / ".archon" / "traces.db"


def _get_store(db: str | None) -> TraceStore:
    path = Path(db) if db else DEFAULT_DB
    if not path.exists():
        click.echo(f"No trace database found at {path}", err=True)
        click.echo("Run an agent first, or specify --db path.", err=True)
        sys.exit(1)
    return TraceStore(path)


@click.group()
@click.version_option(package_name="archon-ai")
def main() -> None:
    """Archon — the production harness for AI agents."""


@main.group()
def traces() -> None:
    """Query and manage agent execution traces."""


@traces.command("list")
@click.option("--db", default=None, help="Path to trace database.")
@click.option("--limit", "-n", default=20, help="Number of runs to show.")
@click.option("--agent", "-a", default=None, help="Filter by agent name.")
@click.option("--json-output", "as_json", is_flag=True, help="Output as JSON.")
def traces_list(db: str | None, limit: int, agent: str | None, as_json: bool) -> None:
    """List recent agent runs."""
    store = _get_store(db)
    runs = store.list_runs(limit=limit, agent=agent)
    store.close()

    if as_json:
        click.echo(json.dumps([{
            "run_id": r.run_id, "agent": r.agent, "steps": r.total_steps,
            "cost": r.total_cost, "tokens": r.total_tokens,
            "latency_ms": r.total_latency_ms, "started_at": r.started_at,
        } for r in runs], indent=2))
        return

    if not runs:
        click.echo("No runs found.")
        return

    # Table header
    click.echo(f"{'RUN ID':<38} {'AGENT':<15} {'STEPS':>5} {'COST':>10} {'TOKENS':>8} {'LATENCY':>10} {'STARTED'}")
    click.echo("─" * 110)
    for r in runs:
        cost_str = f"${r.total_cost:.4f}"
        latency_str = f"{r.total_latency_ms}ms"
        started = r.started_at[:19] if r.started_at else "—"
        click.echo(f"{r.run_id:<38} {r.agent:<15} {r.total_steps:>5} {cost_str:>10} {r.total_tokens:>8} {latency_str:>10} {started}")


@traces.command("show")
@click.argument("run_id")
@click.option("--db", default=None, help="Path to trace database.")
@click.option("--json-output", "as_json", is_flag=True, help="Output as JSON.")
def traces_show(run_id: str, db: str | None, as_json: bool) -> None:
    """Show details of a specific run."""
    store = _get_store(db)
    run = store.get_run(run_id)
    if not run:
        click.echo(f"Run {run_id} not found.", err=True)
        sys.exit(1)

    steps = store.get_steps(run_id)
    audit = store.get_audit(run_id)
    store.close()

    if as_json:
        click.echo(json.dumps({
            "run": {"run_id": run.run_id, "agent": run.agent, "steps": run.total_steps,
                    "cost": run.total_cost, "tokens": run.total_tokens,
                    "latency_ms": run.total_latency_ms, "started_at": run.started_at,
                    "finished_at": run.finished_at},
            "steps": steps,
            "audit": [{"action": a.action, "detail": a.detail, "timestamp": a.timestamp} for a in audit],
        }, indent=2))
        return

    click.echo(f"Run: {run.run_id}")
    click.echo(f"Agent: {run.agent}")
    click.echo(f"Cost: ${run.total_cost:.4f}")
    click.echo(f"Steps: {run.total_steps}")
    click.echo(f"Tokens: {run.total_tokens}")
    click.echo(f"Latency: {run.total_latency_ms}ms")
    click.echo(f"Started: {run.started_at}")
    click.echo(f"Finished: {run.finished_at or '—'}")

    if steps:
        click.echo(f"\n{'#':>3} {'MODEL':<25} {'TIER':<10} {'TOKENS':>8} {'COST':>10} {'LATENCY':>10} {'TOOL'}")
        click.echo("─" * 90)
        for i, s in enumerate(steps):
            tokens = s["input_tokens"] + s["output_tokens"]
            cost_str = f"${s['cost_usd']:.4f}"
            latency_str = f"{s['latency_ms']}ms"
            tool_str = s["tool_call"] or "—"
            cached = " ⚡" if s["cached"] else ""
            click.echo(f"{i+1:>3} {s['model']:<25} {s['tier']:<10} {tokens:>8} {cost_str:>10} {latency_str:>10} {tool_str}{cached}")

    if audit:
        click.echo(f"\nAudit Log:")
        for a in audit:
            detail = f" — {a.detail}" if a.detail else ""
            click.echo(f"  [{a.timestamp[:19]}] {a.action}{detail}")


@traces.command("stats")
@click.option("--db", default=None, help="Path to trace database.")
@click.option("--json-output", "as_json", is_flag=True, help="Output as JSON.")
def traces_stats(db: str | None, as_json: bool) -> None:
    """Show aggregate statistics."""
    store = _get_store(db)
    s = store.stats()
    store.close()

    if as_json:
        click.echo(json.dumps(s, indent=2))
        return

    click.echo(f"Total runs:   {s['total_runs']}")
    click.echo(f"Total cost:   ${s['total_cost_usd']:.4f}")
    click.echo(f"Total tokens: {s['total_tokens']:,}")
    click.echo(f"Total steps:  {s['total_steps']}")

    if s["top_models"]:
        click.echo(f"\nTop Models:")
        for m in s["top_models"]:
            click.echo(f"  {m['model']:<30} {m['calls']:>6} calls  ${m['cost']:.4f}")


@traces.command("purge")
@click.option("--db", default=None, help="Path to trace database.")
@click.option("--before-days", type=int, required=True, help="Delete runs older than N days.")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation.")
def traces_purge(db: str | None, before_days: int, yes: bool) -> None:
    """Delete old traces."""
    from datetime import datetime, timedelta, timezone

    store = _get_store(db)
    cutoff = datetime.now(timezone.utc) - timedelta(days=before_days)

    if not yes:
        click.confirm(f"Delete all traces before {cutoff.date()}?", abort=True)

    count = store.purge_before(cutoff)
    store.close()
    click.echo(f"Purged {count} runs.")
