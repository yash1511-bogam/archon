<p align="center">
  <h1 align="center">Archon</h1>
  <p align="center"><strong>The production harness for AI agents.</strong></p>
  <p align="center">Cost control · Security · Observability · Memory · Evaluation — built-in, not bolted on.</p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/typescript-5.8%2B-blue" alt="TypeScript">
  <img src="https://img.shields.io/badge/rust-2021-orange" alt="Rust">
  <img src="https://img.shields.io/badge/tests-128%20passing-green" alt="Tests">
  <img src="https://img.shields.io/badge/models-140%2B%20registered-blue" alt="Models">
  <img src="https://img.shields.io/badge/providers-17%2B-blue" alt="Providers">
</p>

---

## The Problem

Every AI agent framework — LangChain, CrewAI, LangGraph, Pydantic-AI, OpenAI Agents SDK — solves one problem: **make the agent call tools and get answers.** That's 20% of the work.

The other 80% — cost control, security, observability, memory management, evaluation, governance — every team rebuilds from scratch. That's where **95% of production deployments fail.**

- **84% of companies** take a >6% gross margin hit from uncontrolled AI costs
- **Only 15%** of GenAI deployments have any LLM observability
- **Zero** major frameworks are secure by default (CodeSlick audit: 75–200 critical findings per framework)
- **40% of agentic AI projects** will be canceled by 2027 due to escalating costs (Gartner)

---

## The Solution

Archon ships the production harness as the framework. Not as an add-on. Not as a paid tier. Built-in.

```python
from archon import Agent, tool, Budget

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return web_search(query)

agent = Agent(
    name="researcher",
    instructions="Find accurate information on the given topic.",
    tools=[search],
    model="auto",                        # Router picks cheapest sufficient model per step
    budget=Budget(max_per_run=0.50),     # Hard stop at $0.50 — never overspend
)

result = await agent.run("What caused the 2008 financial crisis?")

print(result.output)                     # The answer
print(f"Cost: ${result.cost:.4f}")       # $0.0312
print(f"Steps: {result.step_count}")     # 4
print(f"Trace: {result.trace_url}")      # http://localhost:8080/traces/abc123
```

Every run returns **cost, steps, and a trace URL**. Observability is not optional.

---

## How Archon Is Different

| Problem | Every Other Framework | Archon |
|---------|----------------------|--------|
| **Cost** | Discover the bill at month-end | Auto-routes to cheapest sufficient model. Hard budget caps. Per-step tracking. **60–70% savings.** |
| **Security** | `exec(llm_output)` in your process | Sandboxed by default. Pre-execution policy checks. Tool output sanitization. Default-deny. |
| **Observability** | "Integrate LangSmith ($39/seat/mo)" | Built-in trace store + web dashboard. Every step logged. Zero external tools. |
| **Memory** | "Here's a vector store adapter" | Tiered memory (working/episodic/semantic/procedural). Temporal decay. Auto-consolidation. |
| **Evaluation** | "Write tests, hope for the best" | Inline verification + async quality scoring + regression detection. Continuous. |
| **Governance** | No audit trails, no compliance | Event sourcing, RBAC, GDPR-compliant erasure. Immutable audit log. |
| **Language** | 9 of 11 frameworks are Python-only | Python + TypeScript from day one. |

---

## Features

### The 5-Gate Execution Pipeline

Every LLM call passes through 5 gates — the harness is in the execution loop, not outside it.

```
User Request
     │
     ▼
┌─────────────────────────────────────────────┐
│ Gate 1: POLICY CHECK                        │
│ Is this agent authorized for this tool?     │
│ Does the policy allow these parameters?     │
│ → Block if denied. Log decision.            │
├─────────────────────────────────────────────┤
│ Gate 2: MODEL ROUTING                       │
│ Classify input complexity (zero latency).   │
│ Pick cheapest model that can handle it.     │
│ Downgrade if budget pressure.               │
├─────────────────────────────────────────────┤
│ Gate 3: EXECUTE                             │
│ Call LLM via LiteLLM (140+ models).         │
│ Run tool calls in sandboxed subprocess.     │
│ Enforce timeout + token limits.             │
├─────────────────────────────────────────────┤
│ Gate 4: VALIDATE OUTPUT                     │
│ Schema validation on structured outputs.    │
│ Strip injection patterns from tool results. │
│ Check for loops (repeated similar actions). │
├─────────────────────────────────────────────┤
│ Gate 5: LOG TRACE                           │
│ Record: model, tokens, cost, latency,       │
│ tool call, routing decision, budget state.  │
│ Append to immutable audit log.              │
└─────────────────────────────────────────────┘
     │
     ▼
  AgentResult { output, cost, steps, trace_url }
```

### Automatic Model Routing

Pattern-based complexity classification — zero LLM calls, zero added latency:

| Tier | When | Default Models | Cost/MTok |
|------|------|---------------|-----------|
| **Simple** | Short queries, yes/no, formatting | Gemini 2.5 Flash, GPT-4.1 Nano | $0.10–0.50 |
| **Standard** | Reasoning, code generation, analysis | Claude Sonnet 4.6, GPT-4.1 Mini | $1–3 |
| **Complex** | Multi-step reasoning, architecture | Claude Opus 4.6, o4-mini | $5–25 |

A typical workload (60% simple, 25% standard, 15% complex) saves **60–70%** vs sending everything to a frontier model.

### Budget Enforcement

Not a dashboard. Not an alert. A **hard stop**.

```python
budget = Budget(max_per_run=0.50, max_per_day=10.00, max_per_month=200.00)
```

When the budget runs low, the router automatically downgrades to cheaper models. When it's exceeded, the agent stops gracefully and returns a partial result. No surprise bills.

### Tiered Memory

Four tiers inspired by cognitive science:

| Tier | What It Stores | Lifecycle |
|------|---------------|-----------|
| **Working** | Current task context (800–2K tokens) | Pinned in LLM context window |
| **Episodic** | Past experiences with timestamps | Decays over time (configurable half-life) |
| **Semantic** | Structured facts and relationships | Consolidated, stale entries pruned |
| **Procedural** | Learned tool-use patterns | Grows from successful runs |

```python
from archon import Memory, MemoryType

memory = Memory(decay_half_life_hours=168)  # 7-day half-life
memory.remember("user_pref", "Prefers concise answers", MemoryType.SEMANTIC)
results = memory.recall("user preferences")
```

### Security

Default-deny policy engine + subprocess sandbox:

```python
from archon import SecurityConfig, SecurityPolicy, PolicyRule, PolicyAction

security = SecurityConfig(
    sandbox=True,
    policy=SecurityPolicy(rules=[
        PolicyRule(tool="search_web", action=PolicyAction.ALLOW),
        PolicyRule(tool="send_email", action=PolicyAction.REQUIRE_APPROVAL),
    ]),
)
```

Tool outputs are sanitized before passing to the LLM — 7 threat categories (instruction override, role hijack, prompt extraction, data exfiltration, delimiter injection, encoded payloads, embedded instructions).

### Continuous Evaluation

Three layers, not just pre-deploy tests:

| Layer | When | What It Catches |
|-------|------|----------------|
| **Inline** | Every request | Schema failures, loops, cost overruns, tool inefficiency |
| **Async** | Sampled traffic | Output quality, coherence, completeness |
| **Regression** | Population-level | Score drops, cost spikes, failure rate increases |

### Shadow Deployments

Test new agent versions safely before promoting:

```python
from archon import ShadowRunner

shadow = ShadowRunner(primary=current_agent, candidate=new_agent)
comparison = await shadow.run("user prompt")
print(comparison.recommendation)  # "PROMOTE — candidate scores 0.12 higher, $0.003 cheaper"
```

### Governance

- **Event sourcing** — every agent action is an immutable event. Full replay and audit trail.
- **RBAC** — role-based tool access. Agents get roles; roles define permissions.
- **GDPR compliance** — right to access (data export) and right to erasure per user.

### Web Dashboard

```bash
archon dashboard --port 8080
```

Dark-themed dashboard with stats grid, top models, run list, step-by-step detail, and audit log. Zero external JS dependencies.

### CLI

```bash
archon traces list                    # Recent runs (table or JSON)
archon traces show <run_id>           # Step-by-step detail + audit log
archon traces stats                   # Aggregate stats with top models
archon traces purge --before-days 30  # Delete old traces
```

### MCP & A2A Protocol Support

Connect to any MCP server and use its tools as native Archon tools:

```python
from archon.protocols import MCPClient

client = MCPClient("npx @modelcontextprotocol/server-github")
client.connect()
tools = client.to_archon_tools()  # Convert MCP tools → Archon ToolDef
agent = Agent(tools=tools, ...)
```

Publish Agent Cards for cross-framework discovery via A2A:

```python
from archon.protocols import AgentCard, AgentSkill

card = AgentCard(
    name="researcher",
    description="Finds and summarizes information",
    url="https://example.com/agents/researcher",
    skills=[AgentSkill(id="search", name="Web Search", description="Search the web")],
)
print(card.to_json())  # Serve at /.well-known/agent.json
```

### Multi-Agent Pipelines

Sequential, parallel, and hierarchical orchestration with durable checkpointing:

```python
from archon import Pipeline, PipelineStep, Parallel, CheckpointStore

pipeline = Pipeline(
    steps=[
        Parallel(steps=[                          # Run concurrently
            PipelineStep(agent=market_analyst),
            PipelineStep(agent=tech_analyst),
        ]),
        PipelineStep(                             # Then synthesize
            agent=synthesizer,
            input_fn=lambda ctx: f"Combine: {ctx['market_analyst']} + {ctx['tech_analyst']}",
        ),
    ],
    checkpoint_store=CheckpointStore(),           # Crash recovery
)

result = await pipeline.run("Analyze the AI agent market")
print(f"Total cost: ${result.total_cost_usd:.4f}")
```

If the pipeline crashes mid-run, re-running with the same `pipeline_id` skips completed steps automatically.

---

## Installation

```bash
# Python (uv recommended)
uv add archon-ai

# TypeScript (pnpm recommended)
pnpm add @archon-ai/sdk
```

Set your API keys:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
# Or any of the 17 supported providers
```

---

## Supported Models (140+ across 17 providers)

Archon ships a built-in model registry with pricing for every major model from 2024–2026. **Azure AI Foundry is fully supported — not just Azure OpenAI, but all 11,000+ Foundry models including Claude, Grok, DeepSeek, Llama, Mistral, Phi, Kimi, and GLM.**

### Direct API Providers

| Provider | Key Models | Input $/MTok |
|----------|-----------|-------------|
| **OpenAI** | GPT-5.5, 5.4, 5.4 Pro, o3, o4-mini, GPT-4.1, GPT-OSS-120B | $0.05 – $30 |
| **Anthropic** | Claude Opus 4.7, Sonnet 4.6, Haiku 4.5 | $0.80 – $5 |
| **Google** | Gemini 3.1 Pro, 2.5 Flash, 2.0 Flash-Lite, Gemma 4 | $0.075 – $2 |
| **xAI** | Grok 4.20, 4, 4.1 Fast, Code Fast 1 | $0.20 – $3 |
| **DeepSeek** | V4 Pro/Flash, V3.2, R2, R1 | $0.14 – $1.74 |
| **Meta** | Llama 4 Maverick/Scout, 3.3 70B | $0.05 – $3 |
| **Mistral** | Large 3, Small 4, Nemo, Devstral 2 | $0.02 – $2 |
| **Alibaba** | Qwen 3.6 Plus, 3.5 Plus, 3 235B | $0.00 – $0.46 |
| **Others** | Cohere, AI21, Microsoft Phi, Moonshot Kimi, Amazon Nova, Perplexity, Zhipu GLM | $0.02 – $3 |

### Platform Support

| Platform | What's Available | Usage |
|----------|-----------------|-------|
| **Azure AI Foundry** | 11,000+ models — GPT-5.x, Claude, Grok, DeepSeek, Llama, Mistral, Phi, Kimi, GLM, Cohere, model-router | `azure/` prefix |
| **AWS Bedrock** | Claude, Llama, Mistral, Nova, Cohere, AI21 | `bedrock/` prefix |
| **OpenRouter** | 1,600+ models via single API | `openrouter/` prefix |
| **Groq** | LPU-accelerated Llama, DeepSeek (300–840 tok/s) | `groq/` prefix |
| **Together AI** | Serverless open models | `together_ai/` prefix |
| **Fireworks AI** | Optimized open model inference | `fireworks_ai/` prefix |
| **Cerebras** | Ultra-fast inference (920 tok/s) | `cerebras/` prefix |

```python
from archon.models import list_models, get_model, estimate_cost

# Find cheap models
cheap = list_models(tier="budget", max_input_cost=0.20)

# Estimate cost for a workload
cost = estimate_cost("claude-sonnet-4.6", input_tokens=50_000, output_tokens=2_000)

# Filter by capability
reasoning_models = list_models(is_reasoning=True)
open_models = list_models(is_open_source=True, min_context=1_000_000)
```

---

## Integration with Other Frameworks

Archon works **alongside** existing tools — use what you need:

```python
# With LangChain / LangGraph — use Archon's router and budget
from archon import Router, Budget
decision = Router().route(user_input, Budget(max_per_run=1.00).remaining)

# With CrewAI — wrap crews with budget tracking
budget = Budget(max_per_run=5.00)
budget.check(estimated_cost)  # Before each task
budget.record(actual_cost)    # After each task

# With Pydantic-AI — types are directly compatible (both use Pydantic 2)
from archon.types import AgentResult, Step  # Serialize, validate, compose freely

# With OpenTelemetry — export traces to Datadog, Grafana, Jaeger
# Built-in OTLP export support
```

---

## Architecture

```
archon/
├── crates/archon-core/        # Rust core — types, budget, router, trace store
├── sdks/python/               # Python SDK (uv + Pydantic 2 + LiteLLM)
│   └── src/archon/
│       ├── agent.py           #   Agent with 5-gate execution pipeline
│       ├── models.py          #   140+ model registry with pricing
│       ├── cache.py           #   Semantic cache (TF-IDF, zero deps)
│       ├── sanitize.py        #   Tool output sanitization (7 threat categories)
│       ├── pipeline.py        #   Multi-agent pipelines + checkpointing
│       ├── dashboard.py       #   Built-in web dashboard
│       ├── shadow.py          #   Shadow deployments
│       ├── governance.py      #   Event sourcing, RBAC, GDPR
│       ├── protocols.py       #   MCP client + A2A Agent Cards
│       ├── cli.py             #   archon CLI (traces, dashboard)
│       ├── budget/            #   Budget enforcement
│       ├── router/            #   Pattern-based model routing
│       ├── memory/            #   Tiered memory system
│       ├── security/          #   Sandbox + policy engine
│       ├── trace/             #   SQLite trace store
│       └── eval/              #   Continuous evaluation engine
├── sdks/typescript/           # TypeScript SDK (pnpm + Zod)
│   └── src/
│       ├── agent.ts           #   Full Agent with LLM integration
│       ├── budget.ts          #   Budget enforcement
│       └── router.ts          #   Pattern-based routing
└── proto/                     # Shared protocol (JSON Schema)
```

**Why Rust + Python + TypeScript?**
- **Rust core** — single binary, memory-safe sandboxing, fast trace storage
- **Python SDK** — Pydantic 2 types, LiteLLM for 140+ models, uv for packaging
- **TypeScript SDK** — Zod types, fetch-based LLM calls, pnpm for packaging

---

## Development

```bash
# Prerequisites: Rust 1.75+, Python 3.10+ (uv), Node.js 20+ (pnpm)

make build    # Build all three
make test     # 4 Rust + 114 Python + 10 TypeScript = 128 tests
make lint     # ruff (Python) + tsc --noEmit (TypeScript)
```

---

## Comparison with Other Frameworks

| Feature | LangGraph | CrewAI | Pydantic-AI | OpenAI SDK | **Archon** |
|---------|-----------|--------|-------------|------------|-----------|
| Built-in model routing | ✗ | ✗ | ✗ | ✗ | ✓ |
| Budget enforcement | ✗ | ✗ | ✗ | ✗ | ✓ |
| Per-step cost tracking | ✗ | ✗ | ✗ | ✗ | ✓ |
| Semantic cache | ✗ | ✗ | ✗ | ✗ | ✓ |
| Sandbox by default | ✗ | ✗ | ✗ | ✗ | ✓ |
| Pre-execution policy | ✗ | ✗ | ✗ | ✗ | ✓ |
| Output sanitization | ✗ | ✗ | ✗ | ✗ | ✓ |
| Built-in traces | ✗ (LangSmith $) | ✗ | ✗ | Partial | ✓ |
| Tiered memory | ✗ | ✗ | ✗ | ✗ | ✓ |
| Continuous eval | ✗ | ✗ | Partial | ✗ | ✓ |
| Shadow deployments | ✗ | ✗ | ✗ | ✗ | ✓ |
| Governance / GDPR | ✗ | ✗ | ✗ | ✗ | ✓ |
| MCP + A2A | ✓ | ✓ | ✓ | ✓ | ✓ |
| Durable execution | ✓ | ✗ | ✓ (ext.) | ✗ | ✓ |
| Python + TypeScript | ✗ | ✗ | ✗ | ✗ | ✓ |

Archon doesn't compete on orchestration patterns. LangGraph's graphs, CrewAI's teams, and Pydantic-AI's DX are excellent. Archon competes on the **production harness** — the infrastructure that makes any agent safe, cheap, and observable.

---

## Philosophy

1. **Secure by default** — unsafe mode requires explicit opt-in.
2. **Cost-aware by default** — every LLM call is tracked, budgeted, and routable.
3. **Observable by default** — every step emits structured traces. No external tools required.
4. **Thin orchestration** — business logic in separate functions. The framework coordinates, not imprisons.
5. **Dual-language** — Python and TypeScript as first-class citizens.
6. **Zero-dependency setup** — SQLite for everything. No Redis, no Postgres, no Docker required.

---

## Contributing

Contributions welcome. See [proto/README.md](proto/README.md) for the shared protocol spec.

```bash
make test  # Run the full suite before submitting
```

---

## License

Apache-2.0 — see [LICENSE](LICENSE).
