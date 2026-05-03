/**
 * Agent — the core Archon TypeScript API.
 *
 * Every LLM call passes through 5 gates:
 *   1. Cache check — skip if we've answered this before
 *   2. Model routing — cheapest sufficient model
 *   3. Budget check — hard stop before overspending
 *   4. Execute — LLM call via OpenAI-compatible API
 *   5. Track — record cost, tokens, latency
 *
 * @example
 * ```ts
 * const agent = new Agent({
 *   name: "researcher",
 *   instructions: "Find accurate information.",
 *   model: "auto",
 *   budget: { maxPerRun: 0.50 },
 *   apiKey: process.env.OPENAI_API_KEY,
 * });
 * const result = await agent.run("What is quantum computing?");
 * console.log(result.output, result.totalCostUsd);
 * ```
 */

import { Budget, BudgetExceeded } from "./budget.js";
import { Router } from "./router.js";
import { TelemetryClient } from "./telemetry.js";
import type { AgentResult, BudgetConfig, RoutingDecision, Step, Tier, ToolDef } from "./types.js";

// ── Configuration ─────────────────────────────────────

/** Conservative cost estimate for budget pre-check (USD). */
const ESTIMATED_COST_PER_CALL = 0.01;

/** Default max steps before the agent stops. */
const DEFAULT_MAX_STEPS = 25;

/** Default OpenAI-compatible API base URL. */
const DEFAULT_API_BASE = "https://api.openai.com/v1";

// ── Agent config ──────────────────────────────────────

export interface AgentConfig {
  /** Human-readable agent name (used in traces). */
  name: string;
  /** System prompt — the agent's personality and rules. */
  instructions: string;
  /** Tools the agent can invoke. */
  tools?: ToolDef[];
  /** Model name, or "auto" for automatic routing. */
  model?: string;
  /** Budget with hard spending limits. */
  budget?: BudgetConfig;
  /** Maximum LLM calls before the agent stops. */
  maxSteps?: number;
  /** API key for the LLM provider. */
  apiKey?: string;
  /** OpenAI-compatible API base URL. */
  apiBase?: string;
  /**
   * Cloud telemetry client. Defaults to a new `TelemetryClient` that
   * auto-enables when `ARCHON_API_KEY` is set in the environment. Pass
   * a configured instance to customize the dashboard URL, or `false`
   * to explicitly disable cloud uploads even if the env var is set.
   */
  telemetry?: TelemetryClient | boolean;
}

// ── Agent class ───────────────────────────────────────

export class Agent {
  readonly name: string;
  readonly instructions: string;
  readonly tools: ToolDef[];
  readonly model: string;
  readonly budget: Budget;
  readonly maxSteps: number;
  private readonly router: Router;
  private readonly apiKey: string;
  private readonly apiBase: string;
  private readonly telemetry: TelemetryClient | undefined;

  constructor(config: AgentConfig) {
    this.name = config.name;
    this.instructions = config.instructions;
    this.tools = config.tools ?? [];
    this.model = config.model ?? "auto";
    this.budget = new Budget(config.budget ?? {});
    this.maxSteps = config.maxSteps ?? DEFAULT_MAX_STEPS;
    this.router = new Router();
    this.apiKey = config.apiKey ?? process.env.OPENAI_API_KEY ?? "";
    this.apiBase = config.apiBase ?? DEFAULT_API_BASE;
    this.telemetry = resolveTelemetry(config.telemetry);
  }

  /**
   * Execute the agent loop with full production harness.
   * Returns an AgentResult with output, cost, steps, and trace URL.
   */
  async run(prompt: string): Promise<AgentResult> {
    const runId = crypto.randomUUID();
    const result: AgentResult = {
      runId,
      output: "",
      steps: [],
      totalCostUsd: 0,
      totalInputTokens: 0,
      totalOutputTokens: 0,
      totalLatencyMs: 0,
      modelUsage: {},
      startedAt: new Date(),
    };

    const messages: ChatMessage[] = [
      { role: "system", content: this.instructions },
      { role: "user", content: prompt },
    ];

    const toolSchemas = this.tools.map(toolToSchema);
    const toolMap = new Map(this.tools.map((t) => [t.name, t]));

    for (let stepNum = 0; stepNum < this.maxSteps; stepNum++) {
      // Gate 1: Route to optimal model
      const currentText = stepNum === 0 ? prompt : lastContent(messages);
      const routing = this.resolveModel(currentText);

      // Gate 2: Check budget
      try {
        this.budget.check(ESTIMATED_COST_PER_CALL);
      } catch {
        result.output = `[Budget exceeded after ${stepNum} steps, spent $${this.budget.spent.toFixed(4)}]`;
        break;
      }

      // Gate 3: Call LLM
      const startTime = performance.now();
      const response = await this.callLLM(routing.model, messages, toolSchemas);
      const latencyMs = Math.round(performance.now() - startTime);

      // Gate 4: Track cost
      const inputTokens = response.usage?.prompt_tokens ?? 0;
      const outputTokens = response.usage?.completion_tokens ?? 0;
      const cost = estimateCost(routing.model, inputTokens, outputTokens);
      this.budget.record(cost);

      const step: Step = {
        id: `${runId}-${stepNum}`,
        model: routing.model,
        tier: routing.tier,
        inputTokens,
        outputTokens,
        costUsd: cost,
        latencyMs,
        cached: false,
        timestamp: new Date(),
      };

      const choice = response.choices?.[0];
      const message = choice?.message;

      // Handle tool calls
      if (message?.tool_calls?.length) {
        messages.push(message as ChatMessage);

        for (const tc of message.tool_calls) {
          const fnName = tc.function.name;
          step.toolCall = fnName;

          let toolOutput: string;
          const tool = toolMap.get(fnName);
          if (tool) {
            try {
              const args = JSON.parse(tc.function.arguments);
              const rawResult = tool.fn(args);
              toolOutput = String(rawResult);
            } catch (err) {
              toolOutput = `[Tool error: ${err}]`;
            }
          } else {
            toolOutput = `[Unknown tool: ${fnName}]`;
          }

          messages.push({
            role: "tool",
            tool_call_id: tc.id,
            content: toolOutput,
          } as ChatMessage);
        }

        recordStep(result, step);
        continue;
      }

      // Final response — no tool calls
      recordStep(result, step);
      result.output = message?.content ?? "";
      break;
    }

    result.finishedAt = new Date();
    result.totalLatencyMs = Math.round(
      result.finishedAt.getTime() - result.startedAt.getTime(),
    );

    // Gate 6: cloud telemetry — fire-and-forget, never blocks the caller.
    // Enabled only when ARCHON_API_KEY is set (or an explicit client was
    // passed). All failures are swallowed inside the client.
    if (this.telemetry?.enabled) {
      this.telemetry.uploadRun(result, this.name);
    }

    return result;
  }

  // ── Private helpers ─────────────────────────────────

  private resolveModel(text: string): RoutingDecision {
    if (this.model !== "auto") {
      return { tier: "standard", model: this.model, reason: "explicit model" };
    }
    return this.router.route(text, this.budget.remaining);
  }

  private async callLLM(
    model: string,
    messages: ChatMessage[],
    tools: object[],
  ): Promise<LLMResponse> {
    const body: Record<string, unknown> = { model, messages };
    if (tools.length > 0) {
      body.tools = tools;
    }

    const response = await fetch(`${this.apiBase}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`LLM API error ${response.status}: ${errorText}`);
    }

    return (await response.json()) as LLMResponse;
  }
}

// ── Helper types (OpenAI-compatible) ──────────────────

interface ChatMessage {
  role: string;
  content?: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
}

interface ToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

interface LLMResponse {
  choices?: Array<{
    message: ChatMessage;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
  };
}

// ── Pure helper functions ─────────────────────────────

function toolToSchema(tool: ToolDef): object {
  return {
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: {
        type: "object",
        properties: tool.parameters,
        required: Object.keys(tool.parameters),
      },
    },
  };
}

function lastContent(messages: ChatMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].content) return messages[i].content!;
  }
  return "";
}

function recordStep(result: AgentResult, step: Step): void {
  result.steps.push(step);
  result.totalCostUsd += step.costUsd;
  result.totalInputTokens += step.inputTokens;
  result.totalOutputTokens += step.outputTokens;
  result.modelUsage[step.model] = (result.modelUsage[step.model] ?? 0) + 1;
}

/**
 * Normalize the `telemetry` constructor argument.
 *
 *   - `undefined` (default) — create a client that auto-enables when
 *     `ARCHON_API_KEY` is set; otherwise stays disabled.
 *   - `false` — opt out entirely; return `undefined`.
 *   - `true` — same as `undefined` (explicit opt-in to defaults).
 *   - `TelemetryClient` — use the provided instance.
 */
function resolveTelemetry(
  telemetry: TelemetryClient | boolean | undefined,
): TelemetryClient | undefined {
  if (telemetry === false) return undefined;
  if (telemetry instanceof TelemetryClient) return telemetry;
  // `true` or `undefined` — build a default client. It no-ops when the
  // env var is missing, so this is safe for offline users.
  return new TelemetryClient();
}

/** Rough cost estimate based on model name patterns. */
function estimateCost(
  model: string,
  inputTokens: number,
  outputTokens: number,
): number {
  // Pricing per million tokens (input / output)
  const pricing: Record<string, [number, number]> = {
    "gpt-4.1-nano": [0.10, 0.40],
    "gpt-4.1-mini": [0.40, 1.60],
    "gpt-4o": [2.50, 10.0],
    "gpt-4.1": [2.00, 8.00],
    "claude-haiku": [0.80, 4.00],
    "claude-sonnet": [3.00, 15.0],
    "claude-opus": [5.00, 25.0],
    "gemini-flash": [0.15, 0.60],
    "gemini-pro": [1.25, 5.00],
    "o4-mini": [1.10, 4.40],
    "deepseek-r1": [0.55, 2.19],
    "deepseek": [0.27, 1.10],
  };

  // Find the best matching pricing entry
  const lowerModel = model.toLowerCase();
  for (const [pattern, [inputRate, outputRate]] of Object.entries(pricing)) {
    if (lowerModel.includes(pattern)) {
      return (inputTokens * inputRate + outputTokens * outputRate) / 1_000_000;
    }
  }

  // Default: mid-tier pricing
  return (inputTokens * 2.0 + outputTokens * 8.0) / 1_000_000;
}
