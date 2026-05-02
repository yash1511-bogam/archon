import type { RoutingDecision, Tier } from "./types.js";

const ANALYSIS = new Set([
  "analyze", "compare", "evaluate", "architect", "design", "optimize", "debug", "refactor",
]);
const MATH = new Set(["calculate", "prove", "derive", "equation", "algorithm", "complexity"]);
const MULTI_STEP = ["step 1", "first,", "then,", "finally,", "after that", "next,"];

// Word-boundary regex — prevents "designer" matching "design".
// Allows verb suffixes (ed, es, ing, tion, ation, ize, ized).
const ANALYSIS_RE = new RegExp(
  `\\b(?:${[...ANALYSIS].join("|")})(?:e?d|e?s|ing|tion|ation|ize|ized)?\\b`,
  "gi",
);
const MATH_RE = new RegExp(
  `\\b(?:${[...MATH].join("|")})(?:e?d|e?s|ing|tion|ation|ize|ized)?\\b`,
  "gi",
);

const DEFAULT_MODELS: Record<Tier, string[]> = {
  simple: ["gemini-2.5-flash", "gpt-4.1-nano"],
  standard: ["claude-sonnet-4.6", "gpt-4.1-mini"],
  complex: ["claude-opus-4.6", "o4-mini"],
};

export class Router {
  private models: Record<Tier, string[]>;

  constructor(models?: Partial<Record<Tier, string[]>>) {
    this.models = { ...DEFAULT_MODELS, ...models };
  }

  route(text: string, remainingBudget?: number): RoutingDecision {
    let tier = this.classify(text);

    if (remainingBudget !== undefined) {
      if (tier === "complex" && remainingBudget < 0.1) tier = "standard";
      if (tier === "standard" && remainingBudget < 0.05) tier = "simple";
    }

    const candidates = this.models[tier];
    return {
      tier,
      model: candidates[0],
      reason: `classified as ${tier}`,
      fallback: candidates[1],
    };
  }

  private classify(text: string): Tier {
    let score = 0;
    const lower = text.toLowerCase();
    const words = lower.split(/\s+/);

    if (words.length > 200) score += 3;
    else if (words.length > 50) score += 1;

    if (text.includes("```")) score += 2;
    if (/\b(def |function |class )\b/.test(text)) score += 2;

    for (const kw of MULTI_STEP) {
      if (lower.includes(kw)) score += 1;
    }
    // Word-boundary matching via regex
    ANALYSIS_RE.lastIndex = 0;
    score += (lower.match(ANALYSIS_RE) ?? []).length * 2;
    MATH_RE.lastIndex = 0;
    score += (lower.match(MATH_RE) ?? []).length * 2;

    if (score <= 2) return "simple";
    if (score <= 6) return "standard";
    return "complex";
  }
}
