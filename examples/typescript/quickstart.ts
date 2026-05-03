/**
 * Minimal example: run an agent and optionally upload telemetry.
 *
 * Usage:
 *   export OPENAI_API_KEY="sk-..."        # your LLM provider key
 *   export ARCHON_API_KEY="arc_..."       # optional — dashboard telemetry
 *
 *   npx tsx examples/typescript/quickstart.ts
 *
 * Without ARCHON_API_KEY the agent runs 100% locally. With it set,
 * every completed run is uploaded to the Archon dashboard.
 */

import { Agent } from "@archon-ai/sdk";

const multiply = {
  name: "multiply",
  description: "Multiply two numbers.",
  fn: (args: unknown) => {
    const { a, b } = args as { a: number; b: number };
    return a * b;
  },
  parameters: {
    a: { type: "number" },
    b: { type: "number" },
  },
};

async function main() {
  const agent = new Agent({
    name: "math-helper",
    instructions: "Use the multiply tool to answer arithmetic questions.",
    tools: [multiply],
    model: "gpt-4.1-mini",
    budget: { maxPerRun: 0.05 },
    // telemetry: false,  // uncomment to opt out even when ARCHON_API_KEY is set
  });

  const result = await agent.run("What is 17 * 23?");
  console.log(result.output);
  console.log(
    `Cost: $${result.totalCostUsd.toFixed(6)}  |  Steps: ${result.steps.length}`,
  );
}

void main();
