# Archon — GitHub Copilot Instructions

Archon is a production harness for AI agents (Rust + Python + TypeScript).

## Conventions

- Python 3.10+, strict typing, Pydantic 2 for models, ruff for linting
- TypeScript 5.8+ strict, Zod for validation, Vitest for tests
- Named constants for all thresholds and limits — no magic numbers
- Functions under 20 lines, single responsibility
- Docstrings on every public class and method

## Style

- `from __future__ import annotations` in every Python file
- Snake_case for Python, camelCase for TypeScript
- Prefer early returns over nested conditionals
- Comments explain WHY, not WHAT — code should be self-documenting

## Do Not

- Use `any` in TypeScript — use `unknown` with type guards
- Use `exec()` or `eval()` on untrusted input
- Add dependencies without justification — zero-dep is a design goal
- Write tests that call real LLM APIs — use MockAgent
- Use `print()` — use structured logging
- Modify `proto/schema.json` without updating all implementations

## Testing

- pytest with asyncio for Python, Vitest for TypeScript
- In-memory SQLite (`None` as db_path) for test isolation
- All 126 tests must pass: `make test`

## Key Files

- `sdks/python/src/archon/agent.py` — core Agent class
- `sdks/python/src/archon/models.py` — 140+ model registry
- `sdks/python/src/archon/router/__init__.py` — pattern-based model routing
- `sdks/typescript/src/agent.ts` — TypeScript Agent class
- `proto/schema.json` — shared type definitions
