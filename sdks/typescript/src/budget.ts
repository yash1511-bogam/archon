import type { BudgetConfig } from "./types.js";

export class BudgetExceeded extends Error {
  constructor(
    public spent: number,
    public limit: number,
    public scope: string = "run",
  ) {
    super(
      `Budget exceeded (${scope}): spent $${spent.toFixed(4)} of $${limit.toFixed(4)} limit`,
    );
    this.name = "BudgetExceeded";
  }
}

export class Budget {
  private runSpent = 0;
  private daySpent = 0;
  private monthSpent = 0;
  private currentDay: number;
  private currentMonth: number;
  private readonly config: Required<BudgetConfig>;

  constructor(config: BudgetConfig = {}) {
    this.config = {
      maxPerRun: config.maxPerRun ?? Infinity,
      maxPerDay: config.maxPerDay ?? Infinity,
      maxPerMonth: config.maxPerMonth ?? Infinity,
      warnAt: config.warnAt ?? 0.8,
    };
    const now = new Date();
    this.currentDay = now.getUTCDate();
    this.currentMonth = now.getUTCMonth();
  }

  check(proposedCost: number): void {
    this.resetIfNewPeriod();

    if (this.runSpent + proposedCost > this.config.maxPerRun) {
      throw new BudgetExceeded(this.runSpent, this.config.maxPerRun, "run");
    }
    if (this.daySpent + proposedCost > this.config.maxPerDay) {
      throw new BudgetExceeded(this.daySpent, this.config.maxPerDay, "day");
    }
    if (this.monthSpent + proposedCost > this.config.maxPerMonth) {
      throw new BudgetExceeded(
        this.monthSpent,
        this.config.maxPerMonth,
        "month",
      );
    }
  }

  record(cost: number): void {
    this.resetIfNewPeriod();
    this.runSpent += cost;
    this.daySpent += cost;
    this.monthSpent += cost;
  }

  get spent(): number {
    return this.runSpent;
  }

  get remaining(): number | undefined {
    if (this.config.maxPerRun === Infinity) return undefined;
    return Math.max(0, this.config.maxPerRun - this.runSpent);
  }

  private resetIfNewPeriod(): void {
    const now = new Date();
    const day = now.getUTCDate();
    const month = now.getUTCMonth();

    if (day !== this.currentDay) {
      this.daySpent = 0;
      this.currentDay = day;
    }
    if (month !== this.currentMonth) {
      this.monthSpent = 0;
      this.currentMonth = month;
    }
  }
}
