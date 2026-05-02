"""Semantic cache — skip LLM calls for similar queries.

Uses hash-based exact match (free) + cosine similarity on TF-IDF vectors (no external deps).
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


@dataclass
class CacheEntry:
    key: str
    response: str
    model: str
    cost_saved: float
    created_at: float


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _cosine_similarity(a: Counter[str], b: Counter[str]) -> float:
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[k] * b[k] for k in common)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class SemanticCache:
    """Two-tier cache: exact hash match + TF-IDF cosine similarity fallback.

    No external embedding model needed — works offline, zero cost, sub-ms latency.
    """

    def __init__(
        self,
        *,
        db_path: Path | str | None = None,
        similarity_threshold: float = 0.92,
        max_entries: int = 10_000,
        ttl_seconds: int = 3600,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.stats = CacheStats()

        if db_path is None:
            self._conn = sqlite3.connect(":memory:")
        else:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(db_path))

        self._conn.execute("PRAGMA journal_mode=WAL")
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

    def get(self, query: str, model: str) -> CacheEntry | None:
        """Look up a cached response. Exact match first, then semantic similarity."""
        self._evict_expired()

        # Tier 1: exact hash match (free)
        h = self._hash(query, model)
        row = self._conn.execute(
            "SELECT query, response, model, cost_saved, created_at FROM cache WHERE hash = ?",
            (h,),
        ).fetchone()
        if row:
            self.stats.hits += 1
            return CacheEntry(key=h, response=row[1], model=row[2], cost_saved=row[3], created_at=row[4])

        # Tier 2: semantic similarity via TF-IDF cosine
        query_vec = Counter(_tokenize(query))
        if not query_vec:
            self.stats.misses += 1
            return None

        rows = self._conn.execute(
            "SELECT hash, query, response, model, cost_saved, created_at FROM cache WHERE model = ? ORDER BY created_at DESC LIMIT 200",
            (model,),
        ).fetchall()

        best_score = 0.0
        best_row = None
        for row in rows:
            candidate_vec = Counter(_tokenize(row[1]))
            score = _cosine_similarity(query_vec, candidate_vec)
            if score > best_score:
                best_score = score
                best_row = row

        if best_row and best_score >= self.similarity_threshold:
            self.stats.hits += 1
            return CacheEntry(
                key=best_row[0], response=best_row[2], model=best_row[3],
                cost_saved=best_row[4], created_at=best_row[5],
            )

        self.stats.misses += 1
        return None

    def put(self, query: str, model: str, response: str, cost: float) -> None:
        """Store a response in the cache."""
        self._enforce_max_entries()
        h = self._hash(query, model)
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (hash, query, response, model, cost_saved, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (h, query, response, model, cost, time.time()),
        )
        self._conn.commit()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM cache")
        self._conn.commit()
        self.stats = CacheStats()

    def _hash(self, query: str, model: str) -> str:
        return hashlib.sha256(f"{model}:{query.strip().lower()}".encode()).hexdigest()[:32]

    def _evict_expired(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        self._conn.execute("DELETE FROM cache WHERE created_at < ?", (cutoff,))

    def _enforce_max_entries(self) -> None:
        count = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        if count >= self.max_entries:
            self._conn.execute(
                "DELETE FROM cache WHERE hash IN (SELECT hash FROM cache ORDER BY created_at ASC LIMIT ?)",
                (count - self.max_entries + 1,),
            )

    def close(self) -> None:
        self._conn.close()
