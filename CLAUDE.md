# CLAUDE.md — Archon

You are working on Archon, a production harness for AI agents. Rust core + Python SDK + TypeScript SDK.

## Critical Rules

- ALWAYS run `make test` after changes — all 126 tests must pass
- ALWAYS use named constants, never bare magic numbers
- ALWAYS add docstrings to public classes and methods
- ALWAYS use `from __future__ import annotations` in Python files
- NEVER use `exec()` or `eval()` on untrusted input
- NEVER add dependencies without strong justification
- NEVER modify `proto/schema.json` without updating Rust, Python, and TypeScript
- NEVER break backward compatibility in the public API

## Tech Stack

- Rust 2021 edition (crates/archon-core/) — types, budget, router, trace store
- Python 3.10+ (sdks/python/) — uv, Pydantic 2, LiteLLM, ruff, mypy strict
- TypeScript 5.8+ (sdks/typescript/) — pnpm, Zod, Vitest, strict mode
- SQLite with WAL mode — default storage for traces, memory, checkpoints, events
- LiteLLM — all LLM calls (140+ models, 17 providers)

## Commands

```bash
make build          # Build all three
make test           # Run all 126 tests (Rust 4 + Python 112 + TypeScript 10)
make lint           # ruff + tsc --noEmit

# Individual
cargo test --manifest-path crates/archon-core/Cargo.toml --lib
cd sdks/python && uv run python -m pytest tests/ -v
cd sdks/typescript && pnpm test
```

## Project Structure

```
crates/archon-core/src/    → Rust core (types.rs, budget.rs, router.rs, trace.rs)
sdks/python/src/archon/
  ├── agent.py             → Agent class with 5-gate execution pipeline
  ├── models.py            → 140+ model registry with pricing
  ├── cache.py             → Semantic cache (TF-IDF cosine, zero deps)
  ├── sanitize.py          → Tool output sanitization (7 threat categories)
  ├── pipeline.py          → Multi-agent pipelines + checkpointing
  ├── dashboard.py         → Built-in web dashboard (Starlette)
  ├── shadow.py            → Shadow deployments for safe rollouts
  ├── governance.py        → Event sourcing, RBAC, GDPR compliance
  ├── protocols.py         → MCP client + A2A Agent Cards
  ├── cli.py               → archon CLI (traces, dashboard)
  ├── budget/              → Budget enforcement with hard caps
  ├── router/              → Pattern-based model routing (27 signals)
  ├── memory/              → Tiered memory (working/episodic/semantic/procedural)
  ├── security/            → Subprocess sandbox + policy engine
  ├── trace/               → SQLite trace store + audit log
  └── eval/                → Continuous evaluation (inline + async + regression)
sdks/typescript/src/       → Agent, Budget, Router (mirrors Python API)
proto/                     → Shared JSON Schema definitions
```

## Code Patterns

When writing Python for this project:

```python
# Named constants at module level
DEFAULT_MAX_STEPS = 25
BUDGET_WARNING_THRESHOLD = 0.8

# Docstrings on public API
class MyClass:
    """One-line summary.

    Longer description if needed.

    Args:
        name: What this parameter does.
    """

# Early returns over deep nesting
def process(data):
    if not data:
        return None
    if data.is_invalid:
        return Error("invalid")
    return transform(data)

# Type hints everywhere
def calculate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    ...
```

## What NOT to Do

- Don't add FastAPI — the dashboard uses Starlette (lighter)
- Don't add numpy/scipy — TF-IDF cosine similarity is hand-rolled (zero deps)
- Don't add a vector database — sqlite-vec is the planned path
- Don't use `print()` — use `tracing` (Rust) or structured logging
- Don't write tests that call real LLM APIs — use MockAgent or VCR cassettes
- Don't put business logic inside the Agent class — extract into focused modules
