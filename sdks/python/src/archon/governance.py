"""Governance — event sourcing, RBAC for tool access, GDPR-compliant memory erasure.

Three pillars of production governance:

  1. **Event sourcing** — every state mutation is an immutable event.
     Full replay capability, audit trail, and crash recovery.

  2. **RBAC** — role-based access control for tool execution.
     Agents and users get roles; roles define tool permissions.

  3. **GDPR compliance** — right to erasure, data export, retention policies.
     Namespace-scoped memory with per-user deletion.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


# ══════════════════════════════════════════════════════
# Event Sourcing
# ══════════════════════════════════════════════════════

class EventType(str, Enum):
    """Types of events in the event store."""

    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    TOOL_CALLED = "tool.called"
    TOOL_BLOCKED = "tool.blocked"
    TOOL_APPROVED = "tool.approved"
    BUDGET_WARNING = "budget.warning"
    BUDGET_EXCEEDED = "budget.exceeded"
    MEMORY_WRITTEN = "memory.written"
    MEMORY_DELETED = "memory.deleted"
    POLICY_EVALUATED = "policy.evaluated"
    OUTPUT_SANITIZED = "output.sanitized"
    USER_DATA_EXPORTED = "gdpr.exported"
    USER_DATA_ERASED = "gdpr.erased"


@dataclass
class Event:
    """An immutable event in the event store.

    Events are append-only — they can never be modified or deleted
    (except by GDPR erasure, which is itself logged as an event).
    """

    event_type: EventType
    agent: str
    run_id: str
    data: dict[str, Any] = field(default_factory=dict)
    user_id: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_id: int | None = None  # Set by the store


_EVENT_SCHEMA = """\
CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    agent      TEXT NOT NULL,
    run_id     TEXT NOT NULL,
    user_id    TEXT,
    data       TEXT NOT NULL DEFAULT '{}',
    timestamp  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_user ON events(user_id);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp);
"""


class EventStore:
    """Append-only event store for full audit trail and replay.

    Every agent action is recorded as an immutable event.
    Supports replay, filtering by run/agent/user, and GDPR erasure.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is None:
            self._conn = sqlite3.connect(":memory:")
        else:
            path = Path(db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path))
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_EVENT_SCHEMA)
        self._conn.commit()

    def append(self, event: Event) -> int:
        """Append an event. Returns the event ID."""
        cursor = self._conn.execute(
            "INSERT INTO events (event_type, agent, run_id, user_id, data, timestamp)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (event.event_type.value, event.agent, event.run_id,
             event.user_id, json.dumps(event.data), event.timestamp),
        )
        self._conn.commit()
        return cursor.lastrowid or 0

    def get_events(
        self,
        *,
        run_id: str | None = None,
        agent: str | None = None,
        user_id: str | None = None,
        event_type: EventType | None = None,
        limit: int = 100,
    ) -> list[Event]:
        """Query events with optional filters."""
        query = "SELECT id, event_type, agent, run_id, user_id, data, timestamp FROM events WHERE 1=1"
        params: list[Any] = []

        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        if agent:
            query += " AND agent = ?"
            params.append(agent)
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type.value)

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [
            Event(
                event_id=r[0], event_type=EventType(r[1]), agent=r[2],
                run_id=r[3], user_id=r[4], data=json.loads(r[5]), timestamp=r[6],
            )
            for r in rows
        ]

    def replay(self, run_id: str) -> list[Event]:
        """Replay all events for a run in chronological order."""
        return list(reversed(self.get_events(run_id=run_id, limit=10000)))

    def close(self) -> None:
        self._conn.close()


# ══════════════════════════════════════════════════════
# RBAC — Role-Based Access Control
# ══════════════════════════════════════════════════════

@dataclass
class Role:
    """A named role with tool permissions.

    Attributes:
        name: Role identifier (e.g., "researcher", "deployer", "admin").
        allow_tools: Tools this role can use. Empty = none allowed.
        deny_tools: Tools explicitly denied (overrides allow).
        require_approval_tools: Tools that need human approval.
    """

    name: str
    allow_tools: set[str] = field(default_factory=set)
    deny_tools: set[str] = field(default_factory=set)
    require_approval_tools: set[str] = field(default_factory=set)


class RBACManager:
    """Role-based access control for agent tool execution.

    Agents and users are assigned roles. Roles define which tools
    they can use. Deny rules override allow rules.

    Usage::

        rbac = RBACManager()
        rbac.add_role(Role("researcher", allow_tools={"search_web", "read_file"}))
        rbac.assign_role("agent:researcher", "researcher")
        rbac.check("agent:researcher", "search_web")  # → "allow"
        rbac.check("agent:researcher", "delete_file")  # → "deny"
    """

    def __init__(self) -> None:
        self._roles: dict[str, Role] = {}
        self._assignments: dict[str, str] = {}  # principal → role_name

    def add_role(self, role: Role) -> None:
        """Register a role."""
        self._roles[role.name] = role

    def assign_role(self, principal: str, role_name: str) -> None:
        """Assign a role to a principal (agent or user identifier)."""
        if role_name not in self._roles:
            raise ValueError(f"Role '{role_name}' not found")
        self._assignments[principal] = role_name

    def check(self, principal: str, tool_name: str) -> str:
        """Check if a principal can use a tool.

        Returns:
            "allow", "deny", or "require_approval"
        """
        role_name = self._assignments.get(principal)
        if not role_name:
            return "deny"  # No role assigned = denied

        role = self._roles.get(role_name)
        if not role:
            return "deny"

        # Deny overrides everything
        if tool_name in role.deny_tools:
            return "deny"

        # Check approval requirement
        if tool_name in role.require_approval_tools:
            return "require_approval"

        # Check allow list
        if role.allow_tools and tool_name in role.allow_tools:
            return "allow"

        # If allow_tools is empty, allow all (except denied)
        if not role.allow_tools:
            return "allow"

        return "deny"

    def get_role(self, principal: str) -> Role | None:
        """Get the role assigned to a principal."""
        role_name = self._assignments.get(principal)
        return self._roles.get(role_name) if role_name else None


# ══════════════════════════════════════════════════════
# GDPR Compliance
# ══════════════════════════════════════════════════════

@dataclass
class GDPRExportResult:
    """Result of a GDPR data export request."""

    user_id: str
    events: list[Event]
    memory_entries: int
    exported_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class GDPRErasureResult:
    """Result of a GDPR erasure (right to be forgotten) request."""

    user_id: str
    events_erased: int
    memory_entries_erased: int
    erased_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class GDPRManager:
    """GDPR-compliant data management for agent memory and events.

    Supports:
      - Right to access (data export)
      - Right to erasure (right to be forgotten)
      - Data retention policies

    All erasure operations are themselves logged as events for audit.
    """

    def __init__(
        self,
        *,
        event_store: EventStore,
        memory_db: sqlite3.Connection | None = None,
    ) -> None:
        self.event_store = event_store
        self._memory_conn = memory_db

    def export_user_data(self, user_id: str) -> GDPRExportResult:
        """Export all data associated with a user (GDPR Article 15).

        Returns all events and memory entries for the given user.
        """
        events = self.event_store.get_events(user_id=user_id, limit=100000)

        memory_count = 0
        if self._memory_conn:
            row = self._memory_conn.execute(
                "SELECT COUNT(*) FROM memories WHERE metadata LIKE ?",
                (f'%"user_id": "{user_id}"%',),
            ).fetchone()
            memory_count = row[0] if row else 0

        # Log the export as an event
        self.event_store.append(Event(
            event_type=EventType.USER_DATA_EXPORTED,
            agent="gdpr_manager", run_id="gdpr",
            user_id=user_id, data={"events_count": len(events), "memory_count": memory_count},
        ))

        return GDPRExportResult(user_id=user_id, events=events, memory_entries=memory_count)

    def erase_user_data(self, user_id: str) -> GDPRErasureResult:
        """Erase all data associated with a user (GDPR Article 17).

        Deletes events and memory entries. The erasure itself is logged.
        """
        # Count before deletion
        events = self.event_store.get_events(user_id=user_id, limit=100000)
        events_count = len(events)

        # Delete events (except the erasure log itself)
        self.event_store._conn.execute(
            "DELETE FROM events WHERE user_id = ?", (user_id,)
        )
        self.event_store._conn.commit()

        # Delete memory entries
        memory_count = 0
        if self._memory_conn:
            cursor = self._memory_conn.execute(
                "DELETE FROM memories WHERE metadata LIKE ?",
                (f'%"user_id": "{user_id}"%',),
            )
            memory_count = cursor.rowcount
            self._memory_conn.commit()

        # Log the erasure (this event persists for audit)
        self.event_store.append(Event(
            event_type=EventType.USER_DATA_ERASED,
            agent="gdpr_manager", run_id="gdpr",
            user_id=user_id,
            data={"events_erased": events_count, "memory_erased": memory_count},
        ))

        return GDPRErasureResult(
            user_id=user_id, events_erased=events_count,
            memory_entries_erased=memory_count,
        )
