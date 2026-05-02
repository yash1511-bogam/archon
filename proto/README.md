# Archon Protocol

Shared type definitions and protocol contracts between Rust core, Python SDK, and TypeScript SDK.

## Tier Classification

All three implementations use the same pattern-based complexity classifier with identical signal weights:

| Signal | Score |
|--------|-------|
| Word count > 200 | +3 |
| Word count > 50 | +1 |
| Contains code blocks (```) | +2 |
| Contains `def`/`function`/`class` | +2 |
| Multi-step keywords (step 1, first, then, finally, after that, next) | +1 each |
| Analysis keywords (analyze, compare, evaluate, architect, design, optimize, debug, refactor) | +2 each |
| Math keywords (calculate, prove, derive, equation, algorithm, complexity) | +2 each |

**Tier thresholds:** 0-2 = Simple, 3-6 = Standard, 7+ = Complex

## Budget-Aware Downgrade

| Original Tier | Remaining Budget | Downgraded To |
|--------------|-----------------|---------------|
| Complex | < $0.10 | Standard |
| Standard | < $0.05 | Simple |

## Default Model Tiers

```json
{
  "simple": ["gemini-2.5-flash", "gpt-4.1-nano"],
  "standard": ["claude-sonnet-4.6", "gpt-4.1-mini"],
  "complex": ["claude-opus-4.6", "o4-mini"]
}
```

## Execution Pipeline

Every LLM call passes through 5 gates:

```
[1. Policy Check] → [2. Route Model] → [3. Execute] → [4. Validate Output] → [5. Log Trace]
```

## Trace Schema

Each step emits a structured trace record:

```json
{
  "id": "run-uuid-step-0",
  "run_id": "uuid",
  "model": "gemini-2.5-flash",
  "tier": "simple",
  "input_tokens": 150,
  "output_tokens": 50,
  "cost_usd": 0.0001,
  "latency_ms": 200,
  "tool_call": null,
  "cached": false,
  "timestamp": "2026-05-02T16:00:00Z"
}
```
