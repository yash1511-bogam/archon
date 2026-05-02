---
description: Archon project conventions
alwaysApply: true
---

# Archon — Cursor Rules

Production harness for AI agents. Rust core + Python SDK (uv) + TypeScript SDK (pnpm).

## Critical

- Run `make test` after changes — 126 tests must pass
- Named constants for all magic numbers
- Docstrings on every public class and method
- `from __future__ import annotations` in Python files
- No `exec()`/`eval()` on untrusted input
- No new dependencies without justification

## Python Style

- Python 3.10+, Pydantic 2, ruff (line-length 100), mypy strict
- Functions under 20 lines, early returns, type hints everywhere
- Module layout: docstring → constants → types → main class → helpers

## TypeScript Style

- TypeScript 5.8+ strict, Zod, Vitest
- No `any` — use `unknown` with type guards

## Testing

- pytest + asyncio (Python), Vitest (TypeScript)
- In-memory SQLite for isolation
- MockAgent for pipeline tests — no real LLM calls

## Don't

- Add FastAPI (use Starlette), numpy/scipy (hand-rolled math), vector DBs (use sqlite-vec)
- Use `print()` — use structured logging
- Modify `proto/schema.json` without updating all three implementations
