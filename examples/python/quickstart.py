"""Minimal example: run an agent and optionally upload telemetry.

Usage:
    export OPENAI_API_KEY="sk-..."        # your LLM provider key
    export ARCHON_API_KEY="arc_..."       # optional — dashboard telemetry

    python examples/python/quickstart.py

Without ARCHON_API_KEY the agent runs 100% locally (SQLite trace store).
With it set, every completed run is uploaded to the Archon dashboard.
"""

from __future__ import annotations

import asyncio

from archon import Agent, Budget, tool


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


async def main() -> None:
    agent = Agent(
        name="math-helper",
        instructions="Use the multiply tool to answer arithmetic questions.",
        tools=[multiply],
        model="gpt-4.1-mini",
        budget=Budget(max_per_run=0.05),
        # telemetry=False,   # uncomment to opt out even when ARCHON_API_KEY is set
    )

    result = await agent.run("What is 17 * 23?")
    print(result.output)
    print(f"Cost: ${result.cost:.6f}  |  Steps: {result.step_count}")
    print(f"Trace: {result.trace_url}")


if __name__ == "__main__":
    asyncio.run(main())
