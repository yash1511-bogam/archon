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
  private currentDay: string;
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
    this.currentDay = toDateKey(now);
    this.currentMonth = now.getUTCMonth() + now.getUTCFullYear() * 12;
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
    this.resetIfNewPeriod();
    const limits: number[] = [];
    if (this.config.maxPerRun !== Infinity)
      limits.push(this.config.maxPerRun - this.runSpent);
    if (this.config.maxPerDay !== Infinity)
      limits.push(this.config.maxPerDay - this.daySpent);
    if (this.config.maxPerMonth !== Infinity)
      limits.push(this.config.maxPerMonth - this.monthSpent);
    if (limits.length === 0) return undefined;
    return Math.max(0, Math.min(...limits));
  }

  private resetIfNewPeriod(): void {
    const now = new Date();
    const day = toDateKey(now);
    const month = now.getUTCMonth() + now.getUTCFullYear() * 12;

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

/** YYYY-MM-DD string — unique per calendar day, no month-boundary collisions. */
function toDateKey(d: Date): string {
  return d.toISOString().slice(0, 10);
}
