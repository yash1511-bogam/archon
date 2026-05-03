<p align="center">
  <h1 align="center">Archon</h1>
  <p align="center"><strong>The production harness for AI agents.</strong></p>
</p>

<p align="center">
  <a href="https://github.com/yash1511-bogam/archon/actions/workflows/ci.yml"><img src="https://github.com/yash1511-bogam/archon/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/archon-framework/"><img src="https://img.shields.io/pypi/v/archon-framework" alt="PyPI"></a>
  <a href="https://www.npmjs.com/package/@archon-ai/sdk"><img src="https://img.shields.io/npm/v/@archon-ai/sdk" alt="npm"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License"></a>
</p>

---

Archon handles the stuff nobody wants to build twice: cost control, security, observability, memory, evaluation, and governance for AI agents. It wraps around your LLM calls — not the other way around.

```bash
pip install archon-framework    # Python
npm install @archon-ai/sdk      # TypeScript
```

```python
from archon import Agent, tool, Budget

@tool
def search(query: str) -> str:
    """Search the web."""
    return web_search(query)

agent = Agent(
    name="researcher",
    instructions="Find accurate information on the given topic.",
    tools=[search],
    model="auto",                     # routes to cheapest model that can handle the task
    budget=Budget(max_per_run=0.50),  # hard stop — not a suggestion
)

result = await agent.run("What caused the 2008 financial crisis?")
print(result.output)
print(f"${result.cost:.4f} across {result.step_count} steps")
```

Every call returns cost, step count, and a trace URL. There's no "add observability later" step.

---

## Why this exists

Every agent framework solves the same 20% of the problem: make the LLM call tools and return answers. The other 80% — budgets, sandboxing, tracing, memory management, eval — gets rebuilt from scratch by every team that ships agents to production.

Archon ships that 80% as the framework itself.

| | Other frameworks | Archon |
|---|---|---|
| **Cost** | Find out at month-end | Per-step tracking, auto-routing to cheapest sufficient model, hard budget caps |
| **Security** | `exec(llm_output)` in your process | Subprocess sandbox, default-deny policy, output sanitization |
| **Observability** | "Integrate LangSmith" | Built-in trace store + dashboard, zero external deps |
| **Memory** | "Here's a vector store adapter" | Four-tier memory with temporal decay and auto-consolidation |
| **Eval** | "Write tests before deploy" | Inline + async + regression detection, continuous |
| **Governance** | Nothing | Event sourcing, RBAC, GDPR erasure |

---

## How it works

Every LLM call passes through five gates:

```
Request → Policy Check → Model Routing → Execute → Validate Output → Log Trace → Result
```

**Policy check** — is this tool allowed? Are the args within bounds? Block or approve.

**Model routing** — classify complexity using 27 lexical signals (no LLM call, sub-ms). Pick the cheapest model for the tier. Downgrade automatically when budget runs low.

**Execute** — call the LLM via LiteLLM (140+ models). Run tool calls in a sandboxed subprocess with timeout.

**Validate** — schema check on structured output. Strip injection patterns. Detect loops.

**Log trace** — record model, tokens, cost, latency, tool calls, routing decision. Append to immutable audit log.

You get back an `AgentResult` with the output, total cost, steps, and a trace URL. Always. Even if the LLM errors out mid-run.

---

## Routing

The router classifies every input and picks the cheapest model that can handle it:

| Tier | Use case | Default models | $/MTok |
|------|----------|---------------|--------|
| Simple | Short queries, formatting, yes/no | Gemini 2.5 Flash, GPT-4.1 Nano | $0.10–0.50 |
| Standard | Reasoning, code gen, analysis | Claude Sonnet 4.6, GPT-4.1 Mini | $1–3 |
| Complex | Multi-step reasoning, architecture | Claude Opus 4.6, o4-mini | $5–25 |

Typical workload (60/25/15 split) saves 60–70% vs sending everything to a frontier model. You can override tiers, pin a specific model, or let the router handle it.

---

## Budget

```python
Budget(max_per_run=0.50, max_per_day=10.00, max_per_month=200.00)
```

All three limits enforced independently. Run budget resets between calls. Day/month reset on calendar boundaries. When any limit gets tight, the router downgrades. When exceeded, the agent stops and returns what it has. No surprise bills.

---

## Security

Default-deny policy engine. Subprocess sandbox for tool execution. Seven-category output sanitizer (instruction override, role hijack, prompt extraction, data exfil, delimiter injection, encoded payloads, embedded instructions).

```python
SecurityConfig(
    sandbox=True,
    policy=SecurityPolicy(rules=[
        PolicyRule(tool="search_web", action=PolicyAction.ALLOW),
        PolicyRule(tool="send_email", action=PolicyAction.REQUIRE_APPROVAL),
    ]),
)
```

---

## Memory

Four tiers: working (pinned in context), episodic (decays over time), semantic (structured facts), procedural (learned tool patterns). Temporal decay with configurable half-life. Auto-consolidation prunes stale entries.

```python
memory = Memory(decay_half_life_hours=168)
memory.remember("user_pref", "Prefers concise answers", MemoryType.SEMANTIC)
results = memory.recall("user preferences")
```

---

## Eval

Three layers running at different points:

- **Inline** (every request) — schema validation, loop detection, cost guards
- **Async** (sampled) — output quality, coherence scoring
- **Regression** (population) — score drops, cost spikes, failure rate changes

Shadow deployments let you run a candidate agent alongside production and compare scores before promoting.

---

## Pipelines

Sequential, parallel, and hierarchical orchestration with SQLite-backed checkpointing. If a pipeline crashes mid-run, re-running skips completed steps.

```python
pipeline = Pipeline(
    steps=[
        Parallel(steps=[PipelineStep(agent=analyst_1), PipelineStep(agent=analyst_2)]),
        PipelineStep(agent=synthesizer, input_fn=lambda ctx: f"Combine: {ctx['analyst_1']} + {ctx['analyst_2']}"),
    ],
    checkpoint_store=CheckpointStore(),
)
result = await pipeline.run("Analyze the AI agent market")
```

---

## Protocols

**MCP** — connect to any MCP server, use its tools as native Archon tools:

```python
client = MCPClient("npx @modelcontextprotocol/server-github")
client.connect()
agent = Agent(tools=client.to_archon_tools(), ...)
```

**A2A** — publish Agent Cards for cross-framework discovery at `/.well-known/agent.json`.

---

## Governance

Event sourcing (every action is an immutable event), RBAC (role-based tool permissions), GDPR compliance (data export + right to erasure). Audit trail survives erasure.

---

## Dashboard & CLI

**Web Dashboard** — full-featured webapp for managing your Archon deployment:

```bash
# Self-hosted dashboard (webapp/ directory)
cd webapp && pnpm dev
```

- **API Key Management** — generate keys with scopes (`runs:write`, `runs:read`, `traces:read`) and optional expiry. Keys are SHA-256 hashed — never stored in plain text (Stripe/GitHub pattern).
- **Real-time Analytics** — cost over time, model distribution, tier breakdown, top models. All charts update instantly via Convex reactive subscriptions.
- **Traces** — every agent run recorded immutably. Click any run to inspect the five-gate pipeline step-by-step.
- **Settings** — profile, notification preferences, danger zone.

Authentication via [Clerk](https://clerk.com). Backend powered by [Convex](https://convex.dev) with real-time data sync. All queries use `ctx.auth.getUserIdentity()` — data is scoped per user, enforced at the database level.

**SDK → Dashboard flow:**

```bash
export ARCHON_API_KEY="arc_..."   # from dashboard → API Keys
# Optional — defaults to https://archon.yashbogam.me
# export ARCHON_BASE_URL="https://archon.yashbogam.me"
```

```
Your code → Agent.run() → POST /api/ingest (Bearer arc_...)
  → Next.js proxy forwards to Convex HTTP action
  → Convex validates key (SHA-256 hash lookup)
  → Inserts run + steps linked to your userId
  → Dashboard updates in real-time (<10ms)
```

When `ARCHON_API_KEY` is **not** set, the SDK runs 100% locally — no network traffic, no cloud dependency. Telemetry upload is a fire-and-forget async task: it never blocks `agent.run()` and never crashes the agent on failure. Opt out explicitly by passing `telemetry=False` to `Agent(...)`.

**CLI** — lightweight terminal interface:

```bash
archon dashboard                      # local web UI at localhost:8080
archon traces list                    # recent runs
archon traces show <run_id>           # step-by-step detail
archon traces stats                   # aggregate stats
```

---

## 140+ Models, 17 Providers

Built-in registry with pricing for every major model. The router uses it for cost-aware selection.

OpenAI, Anthropic, Google, xAI, DeepSeek, Meta, Mistral, Alibaba, Cohere, AI21, Microsoft, Moonshot, Amazon, Perplexity, Zhipu. Platforms: Azure AI Foundry (11,000+ models), AWS Bedrock, OpenRouter, Groq, Together AI, Fireworks AI, Cerebras.

```python
from archon.models import list_models, estimate_cost

cheap = list_models(max_input_cost=0.20)
cost = estimate_cost("claude-sonnet-4.6", input_tokens=50_000, output_tokens=2_000)
```

---

## Works with everything else

Archon isn't a replacement for LangGraph or CrewAI — it's the infrastructure layer underneath. Use the router and budget engine inside any framework:

```python
from archon import Router, Budget

decision = Router().route(user_input, Budget(max_per_run=1.00).remaining)
# → use decision.model in your LangChain/CrewAI/Pydantic-AI call
```

---

## Architecture

```
archon/
├── crates/archon-core/     # Rust — types, budget, router, trace store
├── sdks/python/            # Python — Agent, tools, memory, security, eval, governance
├── sdks/typescript/        # TypeScript — Agent, budget, router
├── webapp/                 # Next.js — Dashboard, API keys, analytics, traces
│   ├── convex/             # Convex backend — schema, functions, HTTP actions
│   └── src/                # React frontend — Clerk auth, GSAP animations, Recharts
└── proto/                  # Shared JSON Schema protocol
```

Rust for the performance-critical paths (budget tracking, trace storage, sandboxing). Python SDK built on Pydantic and LiteLLM. TypeScript SDK built on Zod. Webapp built on Next.js 16, Convex, and Clerk. SQLite for local storage — Convex for cloud dashboard.

---

## Development

```bash
make build    # all three
make test     # 128 tests (4 Rust + 114 Python + 10 TypeScript)
make lint     # ruff + tsc
```

Requires Rust 1.85+, Python 3.11+ with [uv](https://docs.astral.sh/uv/), Node.js 22+ with [pnpm](https://pnpm.io/).

---

## Contributing

PRs welcome. See [proto/README.md](proto/README.md) for the shared protocol spec that all implementations follow.

```bash
make test  # green before submitting
```

---

## License

Apache-2.0

