/**
 * Cloud telemetry exporter — uploads agent runs to the Archon dashboard.
 *
 * When `ARCHON_API_KEY` is set (or `apiKey` is passed explicitly), the
 * SDK uploads each completed run to the Archon web dashboard. Local
 * execution and local tracing continue to work unchanged when the env
 * var is missing — this module is a pure additive layer.
 *
 * Design principles:
 *   - **Off by default.** No `ARCHON_API_KEY` → no network traffic.
 *   - **Never crash the agent.** All errors are caught and logged.
 *   - **Fire-and-forget.** Uploads run on a background Promise.
 *     Call `await flush()` before process exit to drain the queue.
 *   - **Exponential backoff.** Transient 5xx / network errors retry
 *     up to three times with jitter. 4xx fails fast.
 *
 * Wire protocol matches `archon-webapp/convex/http.ts` and the Next.js
 * proxy at `archon-webapp/src/app/api/ingest/route.ts`.
 */

import type { AgentResult } from "./types.js";

// ── Configuration constants ───────────────────────────

const DEFAULT_BASE_URL = "https://archon.yashbogam.me";
const INGEST_PATH = "/api/ingest";
const VALIDATE_PATH = "/api/validate";

/** SDK identifier sent with every request. */
const USER_AGENT = "archon-typescript/0.1.0";

/** Retry tuning — conservative so we never hammer the API on outages. */
const MAX_RETRIES = 3;
const INITIAL_BACKOFF_MS = 500;
const BACKOFF_MULTIPLIER = 2;
const MAX_BACKOFF_MS = 8000;
const REQUEST_TIMEOUT_MS = 10_000;

/** Status codes that should not be retried (auth / bad request). */
const NO_RETRY_STATUS = new Set([400, 401, 403, 404, 422]);

// ── Public types ──────────────────────────────────────

export interface TelemetryClientConfig {
  /**
   * The `arc_...` key. If omitted, falls back to `process.env.ARCHON_API_KEY`.
   * If still absent, the client stays disabled and all operations are no-ops.
   */
  apiKey?: string;
  /** Dashboard base URL. Falls back to `ARCHON_BASE_URL`, then the default. */
  baseUrl?: string;
  /** Optional logger. Defaults to `console`. */
  logger?: Pick<Console, "warn" | "debug">;
}

export interface ValidationResult {
  valid: boolean;
  scopes?: readonly string[];
  expiresAt?: number | null;
  error?: string;
}

// ── Public helpers ────────────────────────────────────

/** Pick the API key from the explicit arg or the `ARCHON_API_KEY` env var. */
export function resolveApiKey(explicit?: string): string | undefined {
  const raw = explicit ?? process.env.ARCHON_API_KEY;
  const trimmed = raw?.trim();
  return trimmed ? trimmed : undefined;
}

/** Pick the dashboard base URL: explicit > env > default. */
export function resolveBaseUrl(explicit?: string): string {
  const raw = explicit ?? process.env.ARCHON_BASE_URL ?? DEFAULT_BASE_URL;
  return raw.replace(/\/+$/, "");
}

// ── Telemetry client ──────────────────────────────────

export class TelemetryClient {
  readonly #apiKey: string | undefined;
  readonly #baseUrl: string;
  readonly #logger: Pick<Console, "warn" | "debug">;
  readonly #pending = new Set<Promise<void>>();

  constructor(config: TelemetryClientConfig = {}) {
    this.#apiKey = resolveApiKey(config.apiKey);
    this.#baseUrl = resolveBaseUrl(config.baseUrl);
    this.#logger = config.logger ?? console;
  }

  /** True when an API key is available and uploads will be attempted. */
  get enabled(): boolean {
    return this.#apiKey !== undefined;
  }

  /**
   * Verify the API key via `GET /api/validate`. Never throws — returns
   * a `ValidationResult` with an `error` message on failure.
   */
  async validate(): Promise<ValidationResult> {
    if (!this.#apiKey) {
      return { valid: false, error: "ARCHON_API_KEY is not set" };
    }

    let response: Response;
    try {
      response = await this.#fetchWithTimeout(
        `${this.#baseUrl}${VALIDATE_PATH}`,
        { method: "GET", headers: this.#authHeaders() },
      );
    } catch (err) {
      return { valid: false, error: `Network error: ${(err as Error).message}` };
    }

    let data: unknown;
    try {
      data = await response.json();
    } catch {
      return { valid: false, error: `Invalid response (${response.status})` };
    }

    const d = data as {
      valid?: boolean;
      scopes?: string[];
      expiresAt?: number | null;
      error?: string;
    };
    if (response.status === 200 && d.valid) {
      return { valid: true, scopes: d.scopes ?? [], expiresAt: d.expiresAt ?? null };
    }
    return { valid: false, error: d.error ?? "Unknown error" };
  }

  /**
   * Schedule an upload for `result` without blocking the caller.
   * The HTTP request runs on a background Promise. Call `flush()` to
   * wait for in-flight uploads to drain before exit.
   */
  uploadRun(result: AgentResult, agentName: string): void {
    if (!this.enabled) return;
    const payload = payloadFromResult(result, agentName);
    const task = this.#sendWithRetry(payload).catch((err) => {
      // This should never happen — sendWithRetry already catches
      // everything — but belt-and-suspenders: the agent must not crash
      // because telemetry failed.
      this.#logger.warn("archon telemetry: uncaught upload error", err);
    });
    this.#pending.add(task);
    void task.finally(() => this.#pending.delete(task));
  }

  /** Wait for all in-flight uploads to complete. */
  async flush(): Promise<void> {
    if (this.#pending.size === 0) return;
    await Promise.allSettled([...this.#pending]);
  }

  // ── Private helpers ─────────────────────────────────

  #authHeaders(): Record<string, string> {
    return {
      Authorization: `Bearer ${this.#apiKey}`,
      "Content-Type": "application/json",
      "User-Agent": USER_AGENT,
    };
  }

  async #fetchWithTimeout(url: string, init: RequestInit): Promise<Response> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    try {
      return await fetch(url, { ...init, signal: controller.signal });
    } finally {
      clearTimeout(timeout);
    }
  }

  async #sendWithRetry(payload: IngestPayload): Promise<void> {
    const url = `${this.#baseUrl}${INGEST_PATH}`;
    let backoff = INITIAL_BACKOFF_MS;

    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
      let response: Response;
      try {
        response = await this.#fetchWithTimeout(url, {
          method: "POST",
          headers: this.#authHeaders(),
          body: JSON.stringify(payload),
        });
      } catch (err) {
        this.#logger.debug(
          `archon telemetry network error (attempt ${attempt}): ${(err as Error).message}`,
        );
        if (attempt >= MAX_RETRIES) {
          this.#logger.warn(
            `archon telemetry upload failed after ${attempt} attempts: ${(err as Error).message}`,
          );
          return;
        }
        await sleepWithJitter(backoff);
        backoff = Math.min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF_MS);
        continue;
      }

      if (response.status === 200) return;
      if (NO_RETRY_STATUS.has(response.status)) {
        this.#logger.warn(
          `archon telemetry rejected: ${response.status} ${await safeErrorBody(response)}`,
        );
        return;
      }
      // 5xx or transient — retry.
      this.#logger.debug(
        `archon telemetry retryable status ${response.status} (attempt ${attempt})`,
      );
      if (attempt >= MAX_RETRIES) {
        this.#logger.warn(
          `archon telemetry upload failed after ${attempt} attempts: ${response.status}`,
        );
        return;
      }
      await sleepWithJitter(backoff);
      backoff = Math.min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF_MS);
    }
  }
}

// ── Payload construction ──────────────────────────────

interface IngestStep {
  step: number;
  gate: string;
  action: string;
  result: string;
  duration: number;
  cost: number;
  model?: string;
  tokens?: number;
}

interface IngestPayload {
  runId: string;
  agent: string;
  model: string;
  tier: string;
  steps: IngestStep[];
  totalCost: number;
  totalTokens: number;
  inputTokens: number;
  outputTokens: number;
  latency: number;
  status: string;
  startedAt: string;
}

function payloadFromResult(result: AgentResult, agentName: string): IngestPayload {
  const steps: IngestStep[] = result.steps.map((step, index) => ({
    step: index + 1,
    gate: step.cached ? "cache" : "execute",
    action: step.toolCall ?? "llm_call",
    result: step.cached ? "cached" : "ok",
    duration: step.latencyMs,
    cost: step.costUsd,
    model: step.model,
    tokens: step.inputTokens + step.outputTokens,
  }));

  const firstStep = result.steps[0];
  const firstModel = firstStep?.model ?? "unknown";
  const firstTier = firstStep?.tier ?? "standard";
  const totalTokens = result.totalInputTokens + result.totalOutputTokens;

  // An output string starting with "[" is how the agent reports errors
  // (e.g. "[Budget exceeded ...]"). Treat empty output the same way so
  // the dashboard can flag failed runs.
  const status = result.output && !result.output.startsWith("[") ? "success" : "error";

  return {
    runId: result.runId,
    agent: agentName,
    model: firstModel,
    tier: firstTier,
    steps,
    totalCost: result.totalCostUsd,
    totalTokens,
    inputTokens: result.totalInputTokens,
    outputTokens: result.totalOutputTokens,
    latency: result.totalLatencyMs,
    status,
    startedAt: result.startedAt.toISOString(),
  };
}

// ── Misc utilities ────────────────────────────────────

async function sleepWithJitter(ms: number): Promise<void> {
  // ±20% jitter to avoid thundering-herd retries.
  const jitter = ms * 0.2 * (2 * Math.random() - 1);
  const wait = Math.max(0, ms + jitter);
  await new Promise((resolve) => setTimeout(resolve, wait));
}

async function safeErrorBody(response: Response): Promise<string> {
  try {
    const data = (await response.json()) as { error?: string };
    if (data && typeof data.error === "string") return data.error.slice(0, 200);
  } catch {
    /* fall through */
  }
  try {
    const text = await response.text();
    return text.slice(0, 200);
  } catch {
    return "";
  }
}
