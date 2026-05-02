import { z } from "zod";

export const Tier = z.enum(["simple", "standard", "complex"]);
export type Tier = z.infer<typeof Tier>;

export interface Step {
  id: string;
  model: string;
  tier: Tier;
  inputTokens: number;
  outputTokens: number;
  costUsd: number;
  latencyMs: number;
  toolCall?: string;
  cached: boolean;
  timestamp: Date;
}

export interface AgentResult {
  runId: string;
  output: string;
  steps: Step[];
  totalCostUsd: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  totalLatencyMs: number;
  modelUsage: Record<string, number>;
  startedAt: Date;
  finishedAt?: Date;
}

export interface BudgetConfig {
  maxPerRun?: number;
  maxPerDay?: number;
  maxPerMonth?: number;
  warnAt?: number;
}

export interface RoutingDecision {
  tier: Tier;
  model: string;
  reason: string;
  fallback?: string;
}

export interface ToolDef {
  name: string;
  description: string;
  fn: (...args: unknown[]) => unknown;
  parameters: Record<string, { type: string }>;
}
