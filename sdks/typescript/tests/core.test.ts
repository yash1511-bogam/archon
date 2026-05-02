import { describe, expect, it } from "vitest";
import { Budget, BudgetExceeded, Router } from "../src/index.js";

describe("Router", () => {
  it("routes simple queries to simple tier", () => {
    const router = new Router();
    const decision = router.route("Hello!");
    expect(decision.tier).toBe("simple");
    expect(decision.model).toBe("gemini-2.5-flash");
  });

  it("routes complex queries to complex tier", () => {
    const router = new Router();
    const decision = router.route(
      "Analyze and compare the architecture, then design an optimized solution with code",
    );
    expect(decision.tier).toBe("complex");
  });

  it("downgrades on budget pressure", () => {
    const router = new Router();
    const decision = router.route("Analyze this architecture", 0.03);
    expect(decision.tier).toBe("simple");
  });
});

describe("Budget", () => {
  it("enforces hard limits", () => {
    const budget = new Budget({ maxPerRun: 0.10 });
    budget.check(0.05);
    budget.record(0.05);
    budget.check(0.04);
    budget.record(0.04);
    expect(() => budget.check(0.02)).toThrow(BudgetExceeded);
  });

  it("tracks remaining budget", () => {
    const budget = new Budget({ maxPerRun: 1.0 });
    expect(budget.remaining).toBe(1.0);
    budget.record(0.3);
    expect(budget.remaining).toBeCloseTo(0.7);
  });

  it("returns undefined remaining when no limit", () => {
    const budget = new Budget();
    budget.check(1000);
    expect(budget.remaining).toBeUndefined();
  });
});


// ── Phase 2: Agent class tests ────────────────────────

import { Agent } from "../src/agent.js";

describe("Agent", () => {
  it("creates with default config", () => {
    const agent = new Agent({
      name: "test",
      instructions: "You are helpful.",
    });
    expect(agent.name).toBe("test");
    expect(agent.model).toBe("auto");
    expect(agent.maxSteps).toBe(25);
  });

  it("creates with explicit model", () => {
    const agent = new Agent({
      name: "test",
      instructions: "Help.",
      model: "gpt-4o",
      budget: { maxPerRun: 1.0 },
      maxSteps: 10,
    });
    expect(agent.model).toBe("gpt-4o");
    expect(agent.maxSteps).toBe(10);
    expect(agent.budget.remaining).toBe(1.0);
  });

  it("has tools accessible", () => {
    const searchTool = {
      name: "search",
      description: "Search the web",
      fn: (query: unknown) => `results for ${query}`,
      parameters: { query: { type: "string" } },
    };
    const agent = new Agent({
      name: "test",
      instructions: "Help.",
      tools: [searchTool],
    });
    expect(agent.tools).toHaveLength(1);
    expect(agent.tools[0].name).toBe("search");
  });

  it("budget is enforced on agent", () => {
    const agent = new Agent({
      name: "test",
      instructions: "Help.",
      budget: { maxPerRun: 0.05 },
    });
    agent.budget.record(0.04);
    expect(agent.budget.remaining).toBeCloseTo(0.01);
    expect(() => agent.budget.check(0.02)).toThrow();
  });
});
