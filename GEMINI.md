# GEMINI.md — Archon

Archon is a production harness for AI agents. Rust core + Python SDK (uv) + TypeScript SDK (pnpm). 126 tests, 140+ model registry.

## Commands

- Build all: `make build`
- Test all: `make test` (4 Rust + 112 Python + 10 TypeScript)
- Lint: `make lint`
- Python tests: `cd sdks/python && uv run python -m pytest tests/ -v`
- TypeScript tests: `cd sdks/typescript && pnpm test`
- Rust tests: `cargo test --manifest-path crates/archon-core/Cargo.toml --lib`

## Rules

- Always run tests after changes
- Python: strict typing, Pydantic 2, ruff, named constants, docstrings
- TypeScript: strict mode, Zod, no `any`
- Never use `exec()`/`eval()` on untrusted input
- Never add dependencies without justification
- Never modify `proto/schema.json` without updating all implementations
- Functions under 20 lines, early returns, comments explain WHY not WHAT

## Structure

- `crates/archon-core/` — Rust core (types, budget, router, trace)
- `sdks/python/src/archon/` — Python SDK (agent, models, cache, security, eval, governance, MCP)
- `sdks/typescript/src/` — TypeScript SDK (agent, budget, router)
- `proto/` — Shared JSON Schema protocol
