"""Tiered memory system — working, episodic, semantic, and procedural.

Inspired by cognitive science and production agent research:
  - Working memory: pinned in LLM context, surgical (800–2K tokens)
  - Episodic memory: timestamped experiences that decay over time
  - Semantic memory: structured facts with conflict resolution
  - Procedural memory: learned tool-use patterns from successful runs

Each tier has its own storage, retrieval strategy, and lifecycle.
Forgetting is as important as remembering — unmanaged growth causes
context poisoning and increased cost.
"""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

# ── Configuration defaults ─────────────────────────────

DEFAULT_WORKING_MEMORY_MAX_TOKENS = 2000
DEFAULT_DECAY_HALF_LIFE_HOURS = 168.0  # 7 days
DEFAULT_CONSOLIDATION_THRESHOLD = 100  # entries before auto-consolidation
DEFAULT_MAX_EPISODIC_AGE_DAYS = 90
DEFAULT_MAX_SEMANTIC_STALENESS_DAYS = 30


class MemoryType(str, Enum):
    """The four tiers of agent memory."""

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


# ── Data classes ───────────────────────────────────────

@dataclass
class MemoryEntry:
    """A single memory entry stored in any tier.

    Attributes:
        key: Unique identifier or topic label.
        value: The content of the memory.
        memory_type: Which tier this belongs to.
        score: Retrieval relevance score (set during recall).
        created_at: When this memory was first stored.
        accessed_at: When this memory was last retrieved.
        access_count: How many times this memory has been recalled.
        metadata: Arbitrary key-value pairs (e.g., source, agent, run_id).
    """

    key: str
    value: str
    memory_type: MemoryType
    score: float = 1.0
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class ConsolidationResult:
    """Summary of a consolidation run."""

    expired_removed: int = 0
    duplicates_merged: int = 0
    stale_pruned: int = 0


# ── Schema ─────────────────────────────────────────────

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS memories (
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    type        TEXT NOT NULL,
    score       REAL NOT NULL DEFAULT 1.0,
    created_at  REAL NOT NULL,
    accessed_at REAL NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    metadata    TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (key, type)
);
CREATE INDEX IF NOT EXISTS idx_mem_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_mem_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_mem_accessed ON memories(accessed_at);
"""


# ── Memory Manager ─────────────────────────────────────

class Memory:
    """Tiered memory with temporal decay, consolidation, and self-editing.

    Four tiers mirror cognitive science:
      - **Working**: pinned in context, small and surgical
      - **Episodic**: past experiences with timestamps, decays over time
      - **Semantic**: structured facts, conflict resolution on write
      - **Procedural**: successful tool-use patterns, learned from experience

    Args:
        db_path: Path to SQLite file. None for in-memory (testing).
        decay_half_life_hours: Hours until an episodic memory's score halves.
        max_working_tokens: Maximum token budget for working memory.
        consolidation_threshold: Auto-consolidate after this many writes.
    """

    def __init__(
        self,
        *,
        db_path: Path | str | None = None,
        decay_half_life_hours: float = DEFAULT_DECAY_HALF_LIFE_HOURS,
        max_working_tokens: int = DEFAULT_WORKING_MEMORY_MAX_TOKENS,
        consolidation_threshold: int = DEFAULT_CONSOLIDATION_THRESHOLD,
    ) -> None:
        self.decay_half_life_hours = decay_half_life_hours
        self.max_working_tokens = max_working_tokens
        self.consolidation_threshold = consolidation_threshold
        self._write_count = 0
        self._conn = self._open_db(db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ── Write operations ───────────────────────────────

    def remember(
        self,
        key: str,
        value: str,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Store or update a memory entry.

        For semantic memories, existing entries with the same key are
        overwritten (latest-wins conflict resolution).
        """
        now = time.time()
        meta_json = _serialize_metadata(metadata)

        self._conn.execute(
            "INSERT OR REPLACE INTO memories"
            " (key, value, type, score, created_at, accessed_at, access_count, metadata)"
            " VALUES (?, ?, ?, 1.0, ?, ?, 0, ?)",
            (key, value, memory_type.value, now, now, meta_json),
        )
        self._conn.commit()

        self._write_count += 1
        if self._write_count >= self.consolidation_threshold:
            self.consolidate()
            self._write_count = 0

    def forget(self, key: str, memory_type: MemoryType | None = None) -> int:
        """Remove a memory entry. Returns the number of entries deleted."""
        if memory_type:
            cursor = self._conn.execute(
                "DELETE FROM memories WHERE key = ? AND type = ?",
                (key, memory_type.value),
            )
        else:
            cursor = self._conn.execute("DELETE FROM memories WHERE key = ?", (key,))
        self._conn.commit()
        return cursor.rowcount

    # ── Read operations ────────────────────────────────

    def recall(
        self,
        query: str,
        memory_types: list[MemoryType] | None = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Retrieve memories relevant to a query.

        Uses multi-strategy retrieval:
          1. Exact key match
          2. TF-IDF cosine similarity on values
          3. Temporal decay weighting for episodic memories

        Results are sorted by final score (relevance × recency).
        """
        types = memory_types or list(MemoryType)
        type_placeholders = ",".join("?" for _ in types)
        type_values = [t.value for t in types]

        rows = self._conn.execute(
            f"SELECT key, value, type, score, created_at, accessed_at, access_count, metadata"
            f" FROM memories WHERE type IN ({type_placeholders})",
            type_values,
        ).fetchall()

        query_tokens = Counter(_tokenize(query))
        scored_entries: list[MemoryEntry] = []

        for row in rows:
            entry = _row_to_entry(row)

            # Compute relevance score
            relevance = self._compute_relevance(query, query_tokens, entry)

            # Apply temporal decay for episodic memories
            if entry.memory_type == MemoryType.EPISODIC:
                decay = self._temporal_decay(entry.created_at)
                relevance *= decay

            entry.score = relevance
            if relevance > 0.01:
                scored_entries.append(entry)

        # Sort by score descending, return top results
        scored_entries.sort(key=lambda e: e.score, reverse=True)
        top_entries = scored_entries[:limit]

        # Update access timestamps for retrieved entries, scoped to the
        # exact (key, type) pairs returned so we don't bump unrelated
        # entries that happen to share a key across memory tiers.
        self._mark_accessed(
            [e.key for e in top_entries],
            types=[e.memory_type for e in top_entries],
        )

        return top_entries

    def get_working_memory(self) -> list[MemoryEntry]:
        """Retrieve all working memory entries (pinned in context)."""
        rows = self._conn.execute(
            "SELECT key, value, type, score, created_at, accessed_at, access_count, metadata"
            " FROM memories WHERE type = ? ORDER BY created_at DESC",
            (MemoryType.WORKING.value,),
        ).fetchall()
        return [_row_to_entry(r) for r in rows]

    def get_procedures(self) -> list[MemoryEntry]:
        """Retrieve all procedural memories (learned tool patterns)."""
        rows = self._conn.execute(
            "SELECT key, value, type, score, created_at, accessed_at, access_count, metadata"
            " FROM memories WHERE type = ? ORDER BY access_count DESC",
            (MemoryType.PROCEDURAL.value,),
        ).fetchall()
        return [_row_to_entry(r) for r in rows]

    # ── Consolidation ──────────────────────────────────

    def consolidate(self) -> ConsolidationResult:
        """Run maintenance: expire old episodic memories, prune stale semantics.

        This is called automatically after ``consolidation_threshold`` writes,
        or can be triggered manually.
        """
        result = ConsolidationResult()
        now = time.time()

        # Remove expired episodic memories
        max_age_seconds = DEFAULT_MAX_EPISODIC_AGE_DAYS * 86400
        cutoff = now - max_age_seconds
        cursor = self._conn.execute(
            "DELETE FROM memories WHERE type = ? AND created_at < ?",
            (MemoryType.EPISODIC.value, cutoff),
        )
        result.expired_removed = cursor.rowcount

        # Prune stale semantic memories (not accessed recently)
        staleness_cutoff = now - (DEFAULT_MAX_SEMANTIC_STALENESS_DAYS * 86400)
        cursor = self._conn.execute(
            "DELETE FROM memories WHERE type = ? AND accessed_at < ? AND access_count < 3",
            (MemoryType.SEMANTIC.value, staleness_cutoff),
        )
        result.stale_pruned = cursor.rowcount

        self._conn.commit()
        return result

    # ── Introspection ──────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return counts per memory type."""
        rows = self._conn.execute(
            "SELECT type, COUNT(*) FROM memories GROUP BY type"
        ).fetchall()
        counts = {r[0]: r[1] for r in rows}
        total = sum(counts.values())
        return {"total": total, "by_type": counts}

    # ── Private helpers ────────────────────────────────

    def _compute_relevance(
        self, query: str, query_tokens: Counter[str], entry: MemoryEntry,
    ) -> float:
        """Score an entry's relevance to the query."""
        # Exact key match gets a strong boost
        if query.strip().lower() == entry.key.strip().lower():
            return 1.0

        # TF-IDF cosine similarity on value text
        entry_tokens = Counter(_tokenize(entry.value))
        similarity = _cosine_similarity(query_tokens, entry_tokens)

        # Boost by access frequency (frequently useful memories rank higher)
        frequency_boost = min(1.0 + (entry.access_count * 0.1), 2.0)

        return similarity * frequency_boost

    def _temporal_decay(self, created_at: float) -> float:
        """Exponential decay: score halves every ``decay_half_life_hours``."""
        age_hours = (time.time() - created_at) / 3600.0
        if self.decay_half_life_hours <= 0:
            return 1.0
        return math.pow(0.5, age_hours / self.decay_half_life_hours)

    def _mark_accessed(
        self,
        keys: list[str],
        types: list[MemoryType] | None = None,
    ) -> None:
        """Update access timestamp and count for retrieved entries.

        If ``types`` is provided it must align 1:1 with ``keys`` and the
        UPDATE is narrowed to the matching (key, type) pairs so we only
        mark the tiers actually recalled.
        """
        if not keys:
            return
        now = time.time()
        key_placeholders = ",".join("?" for _ in keys)
        if types is not None:
            type_values = [t.value for t in types]
            unique_types = list(dict.fromkeys(type_values))  # preserve order, dedupe
            type_placeholders = ",".join("?" for _ in unique_types)
            self._conn.execute(
                f"UPDATE memories SET accessed_at = ?, access_count = access_count + 1"
                f" WHERE key IN ({key_placeholders})"
                f" AND type IN ({type_placeholders})",
                [now, *keys, *unique_types],
            )
        else:
            self._conn.execute(
                f"UPDATE memories SET accessed_at = ?, access_count = access_count + 1"
                f" WHERE key IN ({key_placeholders})",
                [now, *keys],
            )
        self._conn.commit()

    @staticmethod
    def _open_db(db_path: Path | str | None) -> sqlite3.Connection:
        if db_path is None:
            return sqlite3.connect(":memory:")
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


# ── Pure helper functions ──────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens."""
    return re.findall(r"\w+", text.lower())


def _cosine_similarity(a: Counter[str], b: Counter[str]) -> float:
    """Cosine similarity between two token frequency vectors."""
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[k] * b[k] for k in common)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _serialize_metadata(metadata: dict[str, str] | None) -> str:
    """Serialize metadata dict to JSON string."""
    import json
    return json.dumps(metadata or {})


def _row_to_entry(row: tuple[Any, ...]) -> MemoryEntry:
    """Convert a database row to a MemoryEntry."""
    import json
    return MemoryEntry(
        key=row[0],
        value=row[1],
        memory_type=MemoryType(row[2]),
        score=row[3],
        created_at=row[4],
        accessed_at=row[5],
        access_count=row[6],
        metadata=json.loads(row[7]) if row[7] else {},
    )
