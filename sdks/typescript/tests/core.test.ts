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
