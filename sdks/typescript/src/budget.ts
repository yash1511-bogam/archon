import type { BudgetConfig } from "./types.js";

export class BudgetExceeded extends Error {
  constructor(
    public spent: number,
    public limit: number,
  ) {
    super(`Budget exceeded: spent $${spent.toFixed(4)} of $${limit.toFixed(4)} limit`);
    this.name = "BudgetExceeded";
  }
}

export class Budget {
  private runSpent = 0;
  private readonly config: Required<BudgetConfig>;

  constructor(config: BudgetConfig = {}) {
    this.config = {
      maxPerRun: config.maxPerRun ?? Infinity,
      maxPerDay: config.maxPerDay ?? Infinity,
      maxPerMonth: config.maxPerMonth ?? Infinity,
      warnAt: config.warnAt ?? 0.8,
    };
  }

  check(proposedCost: number): void {
    if (this.runSpent + proposedCost > this.config.maxPerRun) {
      throw new BudgetExceeded(this.runSpent, this.config.maxPerRun);
    }
  }

  record(cost: number): void {
    this.runSpent += cost;
  }

  get spent(): number {
    return this.runSpent;
  }

  get remaining(): number | undefined {
    if (this.config.maxPerRun === Infinity) return undefined;
    return Math.max(0, this.config.maxPerRun - this.runSpent);
  }
}
