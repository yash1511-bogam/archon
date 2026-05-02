"""Built-in web dashboard — ``archon dashboard`` serves trace data as HTML.

A lightweight FastAPI server that reads from the TraceStore and renders
a production-grade dashboard with zero external JS dependencies.
Runs on ``http://localhost:8080`` by default.

Usage::

    archon dashboard                  # Start on :8080
    archon dashboard --port 9090      # Custom port
    archon dashboard --db ./my.db     # Custom trace database
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from archon.trace import TraceStore

# ── HTML Templates (inline — zero external deps) ──────

_CSS = """\
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#0f1117;color:#e1e4e8;line-height:1.6}
.container{max-width:1200px;margin:0 auto;padding:20px}
h1{font-size:1.8rem;margin-bottom:4px;color:#58a6ff}
.subtitle{color:#8b949e;margin-bottom:24px;font-size:0.9rem}
.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:32px}
.stat-card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px}
.stat-value{font-size:2rem;font-weight:700;color:#58a6ff}
.stat-label{color:#8b949e;font-size:0.85rem;margin-top:4px}
table{width:100%;border-collapse:collapse;margin-top:16px}
th{text-align:left;padding:10px 12px;border-bottom:2px solid #30363d;color:#8b949e;
  font-size:0.8rem;text-transform:uppercase;letter-spacing:0.5px}
td{padding:10px 12px;border-bottom:1px solid #21262d;font-size:0.9rem}
tr:hover{background:#161b22}
.badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:0.75rem;font-weight:600}
.badge-simple{background:#1f3a1f;color:#3fb950}
.badge-standard{background:#1f2d3a;color:#58a6ff}
.badge-complex{background:#3a1f2d;color:#f85149}
.cost{color:#3fb950;font-family:'SF Mono',monospace}
a{color:#58a6ff;text-decoration:none}
a:hover{text-decoration:underline}
.section{margin-top:32px}
.section h2{font-size:1.2rem;margin-bottom:12px;color:#c9d1d9}
.models-bar{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
.model-chip{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:6px 12px;font-size:0.8rem}
.empty{color:#8b949e;text-align:center;padding:40px}
nav{background:#161b22;border-bottom:1px solid #30363d;padding:12px 0;margin-bottom:24px}
nav .container{display:flex;align-items:center;gap:16px}
nav a{color:#c9d1d9;font-size:0.9rem}
nav a.active{color:#58a6ff;font-weight:600}
.back{margin-bottom:16px;display:inline-block}
.step-row td:first-child{font-family:'SF Mono',monospace;color:#8b949e}
.audit-entry{padding:8px 0;border-bottom:1px solid #21262d;font-size:0.85rem}
.audit-time{color:#8b949e;font-family:'SF Mono',monospace;margin-right:8px}
.audit-action{color:#58a6ff;font-weight:600}
"""


def _page(title: str, body: str, nav_active: str = "dashboard") -> str:
    """Wrap body content in the full HTML page shell."""
    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — Archon</title>
<style>{_CSS}</style>
</head><body>
<nav><div class="container">
  <strong style="color:#58a6ff;font-size:1.1rem">⚡ Archon</strong>
  <a href="/" class="{'active' if nav_active == 'dashboard' else ''}">Dashboard</a>
  <a href="/runs" class="{'active' if nav_active == 'runs' else ''}">Runs</a>
</div></nav>
<div class="container">{body}</div>
</body></html>"""


def _tier_badge(tier: str) -> str:
    """Render a colored badge for a complexity tier."""
    return f'<span class="badge badge-{tier}">{tier}</span>'


# ── Dashboard app factory ──────────────────────────────

def create_app(db_path: Path | str | None = None) -> Any:
    """Create the FastAPI dashboard application.

    Args:
        db_path: Path to the trace database. Uses default if None.

    Returns:
        A FastAPI application instance.
    """
    try:
        from starlette.applications import Starlette
        from starlette.responses import HTMLResponse, JSONResponse
        from starlette.routing import Route
    except ImportError:
        raise ImportError(
            "Dashboard requires starlette. Install with: uv pip install starlette uvicorn"
        )

    default_db = Path.home() / ".archon" / "traces.db"
    store_path = Path(db_path) if db_path else default_db

    # Shared store instance — avoids opening a new connection per request
    shared_store = TraceStore(store_path)

    # ── Routes ─────────────────────────────────────────

    async def dashboard(request: Any) -> HTMLResponse:
        stats = shared_store.stats()
        recent = shared_store.list_runs(limit=10)

        stats_html = f"""
        <h1>Dashboard</h1>
        <p class="subtitle">Production observability for your AI agents</p>
        <div class="stats-grid">
          <div class="stat-card">
            <div class="stat-value">{stats['total_runs']}</div>
            <div class="stat-label">Total Runs</div>
          </div>
          <div class="stat-card">
            <div class="stat-value cost">${stats['total_cost_usd']:.4f}</div>
            <div class="stat-label">Total Cost</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{stats['total_tokens']:,}</div>
            <div class="stat-label">Total Tokens</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">{stats['total_steps']}</div>
            <div class="stat-label">Total Steps</div>
          </div>
        </div>"""

        # Top models
        if stats["top_models"]:
            models_html = '<div class="section"><h2>Top Models</h2><div class="models-bar">'
            for m in stats["top_models"]:
                models_html += (
                    f'<div class="model-chip">{m["model"]} '
                    f'<strong>{m["calls"]}</strong> calls '
                    f'<span class="cost">${m["cost"]:.4f}</span></div>'
                )
            models_html += "</div></div>"
        else:
            models_html = ""

        # Recent runs table
        runs_html = _render_runs_table(recent)

        return HTMLResponse(_page(
            "Dashboard",
            stats_html + models_html + f'<div class="section"><h2>Recent Runs</h2>{runs_html}</div>',
        ))

    async def runs_page(request: Any) -> HTMLResponse:
        runs = shared_store.list_runs(limit=100)
        runs_html = _render_runs_table(runs)
        return HTMLResponse(_page("Runs", f"<h1>All Runs</h1>{runs_html}", nav_active="runs"))

    async def run_detail(request: Any) -> HTMLResponse:
        run_id = request.path_params["run_id"]
        run = shared_store.get_run(run_id)
        if not run:
            return HTMLResponse(_page("Not Found", '<p class="empty">Run not found.</p>'), status_code=404)

        steps = shared_store.get_steps(run_id)
        audit = shared_store.get_audit(run_id)

        # Run summary
        body = f"""
        <a href="/runs" class="back">← Back to runs</a>
        <h1>Run {run_id[:12]}…</h1>
        <div class="stats-grid" style="margin-top:16px">
          <div class="stat-card"><div class="stat-value">{run.agent}</div><div class="stat-label">Agent</div></div>
          <div class="stat-card"><div class="stat-value cost">${run.total_cost:.4f}</div><div class="stat-label">Cost</div></div>
          <div class="stat-card"><div class="stat-value">{run.total_steps}</div><div class="stat-label">Steps</div></div>
          <div class="stat-card"><div class="stat-value">{run.total_tokens:,}</div><div class="stat-label">Tokens</div></div>
          <div class="stat-card"><div class="stat-value">{run.total_latency_ms}ms</div><div class="stat-label">Latency</div></div>
        </div>"""

        # Steps table
        if steps:
            body += '<div class="section"><h2>Steps</h2><table><tr><th>#</th><th>Model</th><th>Tier</th><th>Tokens</th><th>Cost</th><th>Latency</th><th>Tool</th></tr>'
            for i, s in enumerate(steps):
                tokens = s["input_tokens"] + s["output_tokens"]
                cache_icon = " ⚡" if s["cached"] else ""
                body += (
                    f'<tr class="step-row"><td>{i+1}</td><td>{s["model"]}</td>'
                    f'<td>{_tier_badge(s["tier"])}</td><td>{tokens:,}</td>'
                    f'<td class="cost">${s["cost_usd"]:.4f}</td><td>{s["latency_ms"]}ms</td>'
                    f'<td>{s["tool_call"] or "—"}{cache_icon}</td></tr>'
                )
            body += "</table></div>"

        # Audit log
        if audit:
            body += '<div class="section"><h2>Audit Log</h2>'
            for a in audit:
                detail = f" — {a.detail}" if a.detail else ""
                body += (
                    f'<div class="audit-entry">'
                    f'<span class="audit-time">{a.timestamp[:19]}</span>'
                    f'<span class="audit-action">{a.action}</span>{detail}</div>'
                )
            body += "</div>"

        return HTMLResponse(_page(f"Run {run_id[:12]}", body))

    async def api_stats(request: Any) -> JSONResponse:
        stats = shared_store.stats()
        return JSONResponse(stats)

    async def api_runs(request: Any) -> JSONResponse:
        runs = shared_store.list_runs(limit=100)
        return JSONResponse([{
            "run_id": r.run_id, "agent": r.agent, "steps": r.total_steps,
            "cost": r.total_cost, "tokens": r.total_tokens,
            "latency_ms": r.total_latency_ms, "started_at": r.started_at,
        } for r in runs])

    app = Starlette(routes=[
        Route("/", dashboard),
        Route("/runs", runs_page),
        Route("/runs/{run_id}", run_detail),
        Route("/api/stats", api_stats),
        Route("/api/runs", api_runs),
    ])
    return app


def _render_runs_table(runs: list[Any]) -> str:
    """Render a list of RunSummary objects as an HTML table."""
    if not runs:
        return '<p class="empty">No runs yet. Run an agent to see data here.</p>'

    html = "<table><tr><th>Run ID</th><th>Agent</th><th>Steps</th><th>Cost</th><th>Tokens</th><th>Latency</th><th>Started</th></tr>"
    for r in runs:
        started = r.started_at[:19] if r.started_at else "—"
        html += (
            f'<tr><td><a href="/runs/{r.run_id}">{r.run_id[:12]}…</a></td>'
            f"<td>{r.agent}</td><td>{r.total_steps}</td>"
            f'<td class="cost">${r.total_cost:.4f}</td>'
            f"<td>{r.total_tokens:,}</td><td>{r.total_latency_ms}ms</td>"
            f"<td>{started}</td></tr>"
        )
    html += "</table>"
    return html


def start_dashboard(*, port: int = 8080, db_path: str | None = None) -> None:
    """Start the dashboard server. Called by ``archon dashboard`` CLI command."""
    try:
        import uvicorn
    except ImportError:
        raise ImportError("Dashboard requires uvicorn. Install with: uv pip install uvicorn")

    app = create_app(db_path)
    print(f"⚡ Archon Dashboard running at http://localhost:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
