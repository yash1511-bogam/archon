"""Cloud telemetry exporter — uploads agent runs to the Archon dashboard.

When ``ARCHON_API_KEY`` is set in the environment (or passed explicitly),
the SDK uploads each completed run to the Archon web dashboard so teams
can review traces, costs, and tier breakdowns without shipping logs.

Design principles:

* **Off by default.** No ``ARCHON_API_KEY`` → no network traffic.
  Local SQLite tracing continues to work unchanged.
* **Never crash the agent.** All network errors are caught and logged.
  A failing dashboard must not break a production agent run.
* **Fire-and-forget.** Uploads happen on a background asyncio task so
  ``Agent.run()`` returns immediately. Callers can ``await`` ``flush()``
  before process exit to drain the queue.
* **Exponential backoff.** Transient 5xx / network errors retry up to
  three times with jitter. 401/403 responses fail fast — a bad key
  won't succeed on retry.

Wire protocol is documented in ``archon-webapp/convex/http.ts`` and the
Next.js proxy at ``archon-webapp/src/app/api/ingest/route.ts``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from archon.types import AgentResult

# ── Configuration constants ───────────────────────────

DEFAULT_BASE_URL = "https://archon.yashbogam.me"
INGEST_PATH = "/api/ingest"
VALIDATE_PATH = "/api/validate"

# Retry tuning — conservative so we never hammer the API on outages.
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 0.5
BACKOFF_MULTIPLIER = 2.0
MAX_BACKOFF_SECONDS = 8.0
REQUEST_TIMEOUT_SECONDS = 10.0

# SDK identifier sent with every request (helps debug version-specific issues).
USER_AGENT = "archon-python/0.2.0"

# Status codes that should not be retried (auth / bad request).
NO_RETRY_STATUS = frozenset({400, 401, 403, 404, 422})

logger = logging.getLogger("archon.telemetry")


# ── Validation result ─────────────────────────────────

@dataclass(frozen=True)
class ValidationResult:
    """Outcome of a ``GET /api/validate`` call."""

    valid: bool
    scopes: tuple[str, ...] = ()
    expires_at: int | None = None
    error: str | None = None


# ── Public helpers ────────────────────────────────────

def resolve_api_key(explicit: str | None = None) -> str | None:
    """Pick the API key from the explicit arg or the ``ARCHON_API_KEY`` env var.

    Returns ``None`` if neither is set — the telemetry exporter treats
    that as "disabled" rather than an error, matching the local-first
    design of the rest of the SDK.
    """
    key = explicit or os.environ.get("ARCHON_API_KEY")
    if not key:
        return None
    key = key.strip()
    return key or None


def resolve_base_url(explicit: str | None = None) -> str:
    """Pick the dashboard base URL, preferring explicit > env > default."""
    url = explicit or os.environ.get("ARCHON_BASE_URL") or DEFAULT_BASE_URL
    return url.rstrip("/")


# ── Telemetry client ──────────────────────────────────

class TelemetryClient:
    """Async client that uploads ``AgentResult`` objects to the dashboard.

    Instantiate once per Agent (or share across agents in the same process).
    The client lazily starts an ``httpx.AsyncClient`` and a background
    task queue on first use, so constructing a ``TelemetryClient`` without
    ever calling ``upload_run`` has zero runtime cost.

    Args:
        api_key: The ``arc_...`` key. If ``None``, the client is disabled
            and all operations become no-ops. This is the default behavior
            when ``ARCHON_API_KEY`` is not set.
        base_url: Dashboard base URL. Defaults to the env var or the
            production hosted dashboard.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
    ) -> None:
        self._api_key = resolve_api_key(api_key)
        self._base_url = resolve_base_url(base_url)
        self._client: httpx.AsyncClient | None = None
        self._pending: set[asyncio.Task[None]] = set()

    @property
    def enabled(self) -> bool:
        """True when an API key is available and uploads will be attempted."""
        return self._api_key is not None

    # ── Validation ─────────────────────────────────────

    async def validate(self) -> ValidationResult:
        """Call ``GET /api/validate`` to verify the API key.

        Safe to call without a key — returns ``ValidationResult(valid=False)``
        with an ``error`` message in that case. Never raises.
        """
        if not self._api_key:
            return ValidationResult(valid=False, error="ARCHON_API_KEY is not set")

        client = self._ensure_client()
        try:
            response = await client.get(
                f"{self._base_url}{VALIDATE_PATH}",
                headers=self._auth_headers(),
            )
        except httpx.HTTPError as exc:
            return ValidationResult(valid=False, error=f"Network error: {exc}")

        try:
            data = response.json()
        except ValueError:
            return ValidationResult(
                valid=False, error=f"Invalid response ({response.status_code})"
            )

        if response.status_code == 200 and data.get("valid"):
            return ValidationResult(
                valid=True,
                scopes=tuple(data.get("scopes", [])),
                expires_at=data.get("expiresAt"),
            )
        return ValidationResult(valid=False, error=data.get("error", "Unknown error"))

    # ── Upload API ─────────────────────────────────────

    def upload_run(
        self,
        result: AgentResult,
        *,
        agent_name: str,
    ) -> None:
        """Schedule an upload for ``result`` without blocking the caller.

        The actual HTTP request happens on a background asyncio task.
        If no event loop is running (sync caller), falls back to a
        one-shot ``asyncio.run`` in a worker thread via
        ``asyncio.get_event_loop().create_task`` semantics.

        This is fire-and-forget by design. Call ``await flush()`` if
        you need to guarantee delivery before process exit.
        """
        if not self.enabled:
            return

        payload = _payload_from_result(result, agent_name=agent_name)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — run synchronously. This path is rare because
            # ``Agent.run`` is already async, but covers sync test harnesses.
            asyncio.run(self._send_with_retry(payload))
            return

        task = loop.create_task(self._send_with_retry(payload))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def flush(self) -> None:
        """Wait for all in-flight uploads to complete."""
        if not self._pending:
            return
        await asyncio.gather(*self._pending, return_exceptions=True)

    async def aclose(self) -> None:
        """Close the underlying HTTP client after flushing pending uploads."""
        await self.flush()
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ── Private helpers ────────────────────────────────

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS)
        return self._client

    async def _send_with_retry(self, payload: dict[str, Any]) -> None:
        """POST ``payload`` to ``/api/ingest`` with bounded retries."""
        client = self._ensure_client()
        url = f"{self._base_url}{INGEST_PATH}"
        backoff = INITIAL_BACKOFF_SECONDS

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await client.post(
                    url, json=payload, headers=self._auth_headers(),
                )
            except httpx.HTTPError as exc:
                logger.debug("archon telemetry network error (attempt %d): %s", attempt, exc)
                if attempt >= MAX_RETRIES:
                    logger.warning(
                        "archon telemetry upload failed after %d attempts: %s", attempt, exc,
                    )
                    return
                await _sleep_with_jitter(backoff)
                backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF_SECONDS)
                continue

            if response.status_code == 200:
                return
            if response.status_code in NO_RETRY_STATUS:
                logger.warning(
                    "archon telemetry rejected: %d %s",
                    response.status_code,
                    _safe_error_body(response),
                )
                return
            # 5xx / other — retry
            logger.debug(
                "archon telemetry retryable status %d (attempt %d)",
                response.status_code, attempt,
            )
            if attempt >= MAX_RETRIES:
                logger.warning(
                    "archon telemetry upload failed after %d attempts: %d",
                    attempt, response.status_code,
                )
                return
            await _sleep_with_jitter(backoff)
            backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF_SECONDS)


# ── Payload construction ──────────────────────────────

def _payload_from_result(result: AgentResult, *, agent_name: str) -> dict[str, Any]:
    """Convert an ``AgentResult`` into the JSON shape the webapp expects.

    Contract defined in ``archon-webapp/convex/http.ts``. Keep in sync.
    """
    steps_payload = []
    for index, step in enumerate(result.steps, start=1):
        steps_payload.append({
            "step": index,
            "gate": "cache" if step.cached else "execute",
            "action": step.tool_call or "llm_call",
            "result": "cached" if step.cached else "ok",
            "duration": step.latency_ms,
            "cost": step.cost_usd,
            "model": step.model,
            "tokens": step.input_tokens + step.output_tokens,
        })

    first_model = result.steps[0].model if result.steps else "unknown"
    first_tier = result.steps[0].tier.value if result.steps else "standard"
    total_tokens = result.total_input_tokens + result.total_output_tokens

    return {
        "runId": result.run_id,
        "agent": agent_name,
        "model": first_model,
        "tier": first_tier,
        "steps": steps_payload,
        "totalCost": result.total_cost_usd,
        "totalTokens": total_tokens,
        "inputTokens": result.total_input_tokens,
        "outputTokens": result.total_output_tokens,
        "latency": result.total_latency_ms,
        "status": "success" if result.output and not result.output.startswith("[") else "error",
        "startedAt": result.started_at.isoformat(),
    }


# ── Misc utilities ────────────────────────────────────

async def _sleep_with_jitter(seconds: float) -> None:
    """Sleep with ±20% jitter to avoid thundering-herd retries."""
    # ``random`` is fine here — this is jitter, not a cryptographic use.
    jitter = seconds * 0.2 * (2.0 * random.random() - 1.0)  # noqa: S311
    await asyncio.sleep(max(0.0, seconds + jitter))


def _safe_error_body(response: httpx.Response) -> str:
    """Extract a short error string from a response without crashing."""
    try:
        data = response.json()
        if isinstance(data, dict) and "error" in data:
            return str(data["error"])[:200]
    except ValueError:
        pass
    return response.text[:200] if response.text else ""
