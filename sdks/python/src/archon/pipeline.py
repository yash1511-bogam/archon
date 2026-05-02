"""Multi-agent pipelines — sequential, parallel, and hierarchical orchestration.

Supports durable execution with SQLite-backed checkpointing:
if a pipeline crashes mid-run, it resumes from the last completed step.

Three orchestration patterns::

    # Sequential: A → B → C
    pipeline = Pipeline(steps=[Step(researcher), Step(writer), Step(editor)])

    # Parallel: A + B + C → merge
    pipeline = Pipeline(steps=[Parallel(analyst_1, analyst_2), Step(synthesizer)])

    # Hierarchical: orchestrator delegates dynamically
    pipeline = Pipeline(steps=[Step(orchestrator, routes={"research": r, "code": c})])
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from archon.types import AgentResult


# ── Step status ────────────────────────────────────────

class StepStatus(str, Enum):
    """Lifecycle state of a pipeline step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ── Agent protocol ─────────────────────────────────────

@runtime_checkable
class Runnable(Protocol):
    """Any object with an async ``run(prompt) -> AgentResult`` method."""

    name: str

    async def run(self, prompt: str) -> AgentResult: ...


# ── Pipeline step definitions ──────────────────────────

@dataclass
class PipelineStep:
    """A single step in a pipeline.

    Attributes:
        agent: The agent to execute.
        input_fn: Transforms pipeline context into the agent's prompt.
            Receives a dict of ``{step_name: output_string}`` from prior steps.
            Defaults to passing the original pipeline task.
        name: Step name (defaults to agent name).
    """

    agent: Runnable
    input_fn: Any | None = None  # Callable[[dict[str, str]], str] | None
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.agent.name

    def build_prompt(self, task: str, context: dict[str, str]) -> str:
        """Build the prompt for this step from the pipeline context."""
        if self.input_fn is not None:
            return self.input_fn(context)
        return task


@dataclass
class Parallel:
    """A group of steps that execute concurrently.

    All steps receive the same context. Results are merged into the
    pipeline context keyed by each step's name.
    """

    steps: list[PipelineStep]
    name: str = "parallel"


# ── Pipeline result ────────────────────────────────────

@dataclass
class PipelineResult:
    """Result of a complete pipeline execution.

    Attributes:
        pipeline_id: Unique identifier for this run.
        outputs: Mapping of step name → output string.
        agent_results: Mapping of step name → full AgentResult.
        total_cost_usd: Sum of all agent costs.
        total_steps: Sum of all agent steps (LLM calls).
        total_latency_ms: Wall-clock time for the pipeline.
        status: Final status of the pipeline.
    """

    pipeline_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    outputs: dict[str, str] = field(default_factory=dict)
    agent_results: dict[str, AgentResult] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    total_steps: int = 0
    total_latency_ms: int = 0
    status: StepStatus = StepStatus.PENDING


# ── Checkpoint store ───────────────────────────────────

_CHECKPOINT_SCHEMA = """\
CREATE TABLE IF NOT EXISTS checkpoints (
    pipeline_id TEXT NOT NULL,
    step_name   TEXT NOT NULL,
    status      TEXT NOT NULL,
    output      TEXT,
    cost_usd    REAL NOT NULL DEFAULT 0,
    steps_count INTEGER NOT NULL DEFAULT 0,
    latency_ms  INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (pipeline_id, step_name)
);
"""


class CheckpointStore:
    """SQLite-backed durable checkpoint store for pipeline recovery.

    If a pipeline crashes, it can resume from the last completed step
    instead of restarting from scratch.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is None:
            self._conn = sqlite3.connect(":memory:")
        else:
            path = Path(db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path))
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_CHECKPOINT_SCHEMA)
        self._conn.commit()

    def save(
        self,
        pipeline_id: str,
        step_name: str,
        status: StepStatus,
        output: str = "",
        cost_usd: float = 0.0,
        steps_count: int = 0,
        latency_ms: int = 0,
    ) -> None:
        """Save or update a checkpoint for a pipeline step."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO checkpoints"
            " (pipeline_id, step_name, status, output, cost_usd, steps_count, latency_ms, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (pipeline_id, step_name, status.value, output, cost_usd, steps_count, latency_ms, now),
        )
        self._conn.commit()

    def get_completed(self, pipeline_id: str) -> dict[str, str]:
        """Return outputs of all completed steps for a pipeline."""
        rows = self._conn.execute(
            "SELECT step_name, output FROM checkpoints"
            " WHERE pipeline_id = ? AND status = ?",
            (pipeline_id, StepStatus.COMPLETED.value),
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def is_completed(self, pipeline_id: str, step_name: str) -> bool:
        """Check if a specific step has already completed."""
        row = self._conn.execute(
            "SELECT 1 FROM checkpoints WHERE pipeline_id = ? AND step_name = ? AND status = ?",
            (pipeline_id, step_name, StepStatus.COMPLETED.value),
        ).fetchone()
        return row is not None

    def close(self) -> None:
        self._conn.close()


# ── Pipeline executor ──────────────────────────────────

class Pipeline:
    """Multi-agent pipeline with sequential, parallel, and hierarchical execution.

    Supports durable checkpointing: if a step fails, the pipeline can be
    re-run with the same ``pipeline_id`` and it will skip completed steps.

    Args:
        steps: Ordered list of PipelineStep or Parallel groups.
        checkpoint_store: Optional checkpoint store for crash recovery.
        pipeline_id: Reuse an existing ID to resume a crashed pipeline.
    """

    def __init__(
        self,
        *,
        steps: list[PipelineStep | Parallel],
        checkpoint_store: CheckpointStore | None = None,
        pipeline_id: str | None = None,
    ) -> None:
        self.steps = steps
        self.checkpoint_store = checkpoint_store
        self.pipeline_id = pipeline_id or str(uuid.uuid4())

    async def run(self, task: str) -> PipelineResult:
        """Execute the pipeline and return aggregated results.

        Steps run in order. Parallel groups run concurrently.
        Completed steps (from checkpoints) are skipped automatically.
        """
        result = PipelineResult(pipeline_id=self.pipeline_id)
        result.status = StepStatus.RUNNING
        start_time = time.monotonic()

        # Load any previously completed steps (crash recovery)
        if self.checkpoint_store:
            result.outputs = self.checkpoint_store.get_completed(self.pipeline_id)

        for step_or_group in self.steps:
            if isinstance(step_or_group, Parallel):
                await self._run_parallel(step_or_group, task, result)
            else:
                await self._run_step(step_or_group, task, result)

        result.total_latency_ms = int((time.monotonic() - start_time) * 1000)
        result.status = StepStatus.COMPLETED
        return result

    async def _run_step(
        self, step: PipelineStep, task: str, result: PipelineResult,
    ) -> None:
        """Execute a single pipeline step, skipping if already checkpointed."""
        # Skip if already completed (crash recovery)
        if step.name in result.outputs:
            return

        if self.checkpoint_store:
            self.checkpoint_store.save(
                self.pipeline_id, step.name, StepStatus.RUNNING,
            )

        prompt = step.build_prompt(task, result.outputs)

        try:
            agent_result = await step.agent.run(prompt)
        except Exception as exc:
            if self.checkpoint_store:
                self.checkpoint_store.save(
                    self.pipeline_id, step.name, StepStatus.FAILED,
                    output=str(exc),
                )
            result.status = StepStatus.FAILED
            raise

        # Record results
        result.outputs[step.name] = agent_result.output
        result.agent_results[step.name] = agent_result
        result.total_cost_usd += agent_result.total_cost_usd
        result.total_steps += agent_result.step_count

        if self.checkpoint_store:
            self.checkpoint_store.save(
                self.pipeline_id, step.name, StepStatus.COMPLETED,
                output=agent_result.output,
                cost_usd=agent_result.total_cost_usd,
                steps_count=agent_result.step_count,
                latency_ms=agent_result.total_latency_ms,
            )

    async def _run_parallel(
        self, group: Parallel, task: str, result: PipelineResult,
    ) -> None:
        """Execute a group of steps concurrently."""
        # Filter out already-completed steps
        pending = [s for s in group.steps if s.name not in result.outputs]
        if not pending:
            return

        tasks = [self._run_step(step, task, result) for step in pending]
        await asyncio.gather(*tasks)
