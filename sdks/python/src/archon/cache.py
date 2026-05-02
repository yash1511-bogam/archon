"""Semantic cache — skip LLM calls for queries we've already answered.

Two-tier lookup:
  1. Exact hash match (free, instant)
  2. TF-IDF cosine similarity (no external deps, sub-ms)

No embedding model needed. Works offline. Zero cost.
"""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# ── Configuration defaults ─────────────────────────────

DEFAULT_SIMILARITY_THRESHOLD = 0.92
DEFAULT_MAX_ENTRIES = 10_000
DEFAULT_TTL_SECONDS = 3600
SEMANTIC_SEARCH_LIMIT = 200  # Max candidates to compare for similarity


@dataclass
class CacheEntry:
    """A cached LLM response."""

    key: str
    response: str
    model: str
    cost_saved: float
    created_at: float


@dataclass
class CacheStats:
    """Running statistics for cache performance monitoring."""

    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups that returned a cached result (0.0–1.0)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class SemanticCache:
    """Two-tier cache: exact hash match + TF-IDF cosine similarity.

    No external embedding model needed — works offline, zero cost,
    sub-millisecond latency.

    Args:
        db_path: Path to SQLite database. None for in-memory (testing).
        similarity_threshold: Minimum cosine similarity for a semantic hit (0.0–1.0).
        max_entries: Maximum cached entries before LRU eviction.
        ttl_seconds: Time-to-live for cache entries.
    """

    def __init__(
        self,
        *,
        db_path: Path | str | None = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.stats = CacheStats()
        self._conn = self._open_db(db_path)
        self._create_tables()

    def get(self, query: str, model: str) -> CacheEntry | None:
        """Look up a cached response. Exact match first, then semantic similarity."""
        self._evict_expired()

        # Tier 1: exact hash match (free)
        entry = self._exact_lookup(query, model)
        if entry:
            self.stats.hits += 1
            return entry

        # Tier 2: semantic similarity via TF-IDF cosine
        entry = self._semantic_lookup(query, model)
        if entry:
            self.stats.hits += 1
            return entry

        self.stats.misses += 1
        return None

    def put(self, query: str, model: str, response: str, cost: float) -> None:
        """Store a response in the cache."""
        self._enforce_max_entries()
        cache_key = self._hash(query, model)
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (hash, query, response, model, cost_saved, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (cache_key, query, response, model, cost, time.time()),
        )
        self._conn.commit()

    def clear(self) -> None:
        """Remove all cached entries and reset statistics."""
        self._conn.execute("DELETE FROM cache")
        self._conn.commit()
        self.stats = CacheStats()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    # ── Private helpers ────────────────────────────────

    def _exact_lookup(self, query: str, model: str) -> CacheEntry | None:
        cache_key = self._hash(query, model)
        row = self._conn.execute(
            "SELECT query, response, model, cost_saved, created_at FROM cache WHERE hash = ?",
            (cache_key,),
        ).fetchone()
        if row:
            return CacheEntry(key=cache_key, response=row[1], model=row[2],
                              cost_saved=row[3], created_at=row[4])
        return None

    def _semantic_lookup(self, query: str, model: str) -> CacheEntry | None:
        query_tokens = Counter(_tokenize(query))
        if not query_tokens:
            return None

        rows = self._conn.execute(
            "SELECT hash, query, response, model, cost_saved, created_at"
            " FROM cache WHERE model = ? ORDER BY created_at DESC LIMIT ?",
            (model, SEMANTIC_SEARCH_LIMIT),
        ).fetchall()

        best_score = 0.0
        best_row = None
        for row in rows:
            candidate_tokens = Counter(_tokenize(row[1]))
            score = _cosine_similarity(query_tokens, candidate_tokens)
            if score > best_score:
                best_score = score
                best_row = row

        if best_row and best_score >= self.similarity_threshold:
            return CacheEntry(key=best_row[0], response=best_row[2], model=best_row[3],
                              cost_saved=best_row[4], created_at=best_row[5])
        return None

    def _evict_expired(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        self._conn.execute("DELETE FROM cache WHERE created_at < ?", (cutoff,))
        self._conn.commit()

    def _enforce_max_entries(self) -> None:
        count = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        if count >= self.max_entries:
            overflow = count - self.max_entries + 1
            self._conn.execute(
                "DELETE FROM cache WHERE hash IN"
                " (SELECT hash FROM cache ORDER BY created_at ASC LIMIT ?)",
                (overflow,),
            )

    @staticmethod
    def _hash(query: str, model: str) -> str:
        normalized = f"{model}:{query.strip().lower()}"
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    @staticmethod
    def _open_db(db_path: Path | str | None) -> sqlite3.Connection:
        if db_path is None:
            return sqlite3.connect(":memory:")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _create_tables(self) -> None:
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS cache (
                hash TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                model TEXT NOT NULL,
                cost_saved REAL NOT NULL DEFAULT 0,
                created_at REAL NOT NULL
            )"""
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_ts ON cache(created_at)")
        self._conn.commit()


# ── Module-level helpers (pure functions) ──────────────

def _tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens."""
    return re.findall(r"\w+", text.lower())


def _cosine_similarity(a: Counter[str], b: Counter[str]) -> float:
    """Compute cosine similarity between two token frequency vectors."""
    common_keys = set(a) & set(b)
    if not common_keys:
        return 0.0
    dot_product = sum(a[k] * b[k] for k in common_keys)
    magnitude_a = math.sqrt(sum(v * v for v in a.values()))
    magnitude_b = math.sqrt(sum(v * v for v in b.values()))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)
