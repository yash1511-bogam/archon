.PHONY: all build test lint clean

all: build test

# ── Build ──────────────────────────────────────────────
build: build-rust build-python build-ts

build-rust:
	cargo build --manifest-path crates/archon-core/Cargo.toml

build-python:
	cd sdks/python && uv sync

build-ts:
	cd sdks/typescript && pnpm install && pnpm build

# ── Test ───────────────────────────────────────────────
test: test-rust test-python test-ts

test-rust:
	cargo test --manifest-path crates/archon-core/Cargo.toml --lib

test-python:
	cd sdks/python && uv run python -m pytest tests/ -v

test-ts:
	cd sdks/typescript && pnpm test

# ── Lint ───────────────────────────────────────────────
lint: lint-python lint-ts

lint-python:
	cd sdks/python && uv run python -m ruff check src/

lint-ts:
	cd sdks/typescript && pnpm typecheck

# ── Clean ──────────────────────────────────────────────
clean:
	cargo clean --manifest-path crates/archon-core/Cargo.toml
	rm -rf sdks/python/.venv sdks/python/dist
	rm -rf sdks/typescript/node_modules sdks/typescript/dist
