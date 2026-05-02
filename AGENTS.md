# Archon — Agent Instructions

Archon is a production harness for AI agents. Rust core + Python SDK (uv) + TypeScript SDK (pnpm). 126 tests, 140+ model registry, Apache-2.0.

## Commands

```bash
# Build
make build                                                    # All three
cargo build --manifest-path crates/archon-core/Cargo.toml     # Rust
cd sdks/python && uv sync                                     # Python
cd sdks/typescript && pnpm install && pnpm build              # TypeScript

# Test
make test                                                     # All (126 tests)
cargo test --manifest-path crates/archon-core/Cargo.toml --lib
cd sdks/python && uv run python -m pytest tests/ -v
cd sdks/typescript && pnpm test

# Lint
cd sdks/python && uv run python -m ruff check src/
cd sdks/typescript && pnpm typecheck
```

## Code Style

### Python (sdks/python/)
- Python 3.10+, strict typing with `from __future__ import annotations`
- Pydantic 2 for all data models, dataclasses for internal types
- Ruff for linting (line-length 100), mypy strict
- Named constants for all magic numbers — never bare `0.10` or `25`
- Functions under 20 lines. Extract helpers with clear names.
- Docstrings on every public class and method (Google style)
- Comments explain WHY, not WHAT
- Module structure: module docstring → constants → types → main class → private helpers

### TypeScript (sdks/typescript/)
- TypeScript 5.8+ strict mode, ESNext target
- Zod for runtime validation, native types for internal use
- Vitest for testing, pnpm for packages
- No `any` — use `unknown` with type guards

### Rust (crates/archon-core/)
- Edition 2021, serde for serialization
- `thiserror` for error types, `tracing` for logging
- Tests in the same file (`#[cfg(test)] mod tests`)

## Architecture

```
crates/archon-core/        → Rust: types, budget, router, trace store
sdks/python/src/archon/    → Python: Agent, tools, memory, security, eval, governance, MCP/A2A
sdks/typescript/src/       → TypeScript: Agent, budget, router (mirrors Python)
proto/                     → Shared JSON Schema protocol definitions
```

The Python SDK is the primary implementation. TypeScript mirrors the core API. Rust provides the performance-critical runtime (budget tracking, trace storage, sandboxing).

## Testing

- Run `make test` before every commit — all 126 tests must pass
- New features require tests. No exceptions.
- Use in-memory SQLite (pass `None` as db_path) for test isolation
- Mock agents with the `MockAgent` pattern from `tests/test_core.py`
- Async tests use `@pytest.mark.asyncio`

## Boundaries

- NEVER modify `proto/schema.json` without updating all three implementations
- NEVER add external dependencies without justification — zero-dep is a design goal
- NEVER use `exec()` or `eval()` on untrusted input
- NEVER store API keys in code or config files
- NEVER break the public API without a deprecation path
- Keep `uv.lock` and `pnpm-lock.yaml` committed — reproducible builds matter

## Key Design Decisions

- **LiteLLM** handles all LLM calls in Python (140+ models). Don't add provider-specific SDKs.
- **SQLite with WAL mode** is the default storage for everything. Don't require Postgres/Redis.
- **Pydantic 2** is the schema layer. Don't introduce marshmallow, attrs, or msgspec.
- **The 5-gate pipeline** (policy → route → execute → validate → trace) is the core loop. Every new feature integrates into one of these gates.
- **Budget enforcement is a hard stop**, not a warning. The agent must never overspend.
