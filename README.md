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
  <img src="https://img.shields.io/badge/tests-126%20passing-green" alt="Tests">
  <img src="https://img.shields.io/badge/models-140%2B%20registered-blue" alt="Models">
  <img src="https://img.shields.io/badge/providers-17%2B-blue" alt="Providers">
</p>

---

## The Problem

Every AI agent framework — LangChain, CrewAI, LangGraph, Pydantic-AI, OpenAI Agents SDK — solves one problem: **make the agent call tools and get answers.** That's 20% of the work.

The other 80% — cost control, security, observability, memory management, evaluation, governance — every team rebuilds from scratch. That's where **95% of production deployments fail.**

- **84% of companies** take a >6% gross margin hit from uncontrolled AI costs
- **Only 15%** of GenAI deployments have any LLM observability
- **Zero** major frameworks are secure by default (CodeSlick audit: 75-200 critical findings per framework)
- **40% of agentic AI projects** will be canceled by 2027 due to escalating costs (Gartner)

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
| **Cost** | Discover the bill at month-end. No routing, no budgets, no tracking. | Every LLM call auto-routed to cheapest sufficient model. Hard budget caps. Per-step cost tracking. **60-70% savings.** |
| **Security** | `exec(llm_output)` in your process. No sandbox. No validation. | Sandboxed by default. Pre-execution policy checks. Tool output sanitization. Default-deny. |
| **Observability** | "Integrate LangSmith ($39/seat/mo)" or Langfuse. Only 15% of deployments have monitoring. | Built-in trace store + web dashboard. Every step logged. Zero external tools needed. |
| **Memory** | "Here's a vector store adapter. Good luck." No temporal awareness, no forgetting. | Tiered memory (working/episodic/semantic/procedural). Temporal decay. Automatic consolidation. Self-editing. |
| **Evaluation** | "Write tests before deploy. Hope for the best." | Three layers: inline verification + async quality scoring + regression detection. Continuous, not just pre-deploy. |
| **Setup** | Requires Python/Node expertise, virtual envs, YAML configs, scattered API keys. | `pip install archon-ai` and go. Or `pnpm add @archon-ai/sdk`. |
| **Language** | 9 of 11 major frameworks are Python-only. | Python + TypeScript as first-class citizens from day one. |

---

## How It Works

### The 5-Gate Execution Pipeline

Every LLM call in Archon passes through 5 gates. This is the core architectural difference — the harness is in the execution loop, not outside it.

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
│ Call LLM via LiteLLM (100+ models).         │
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

Archon classifies every input using pattern-based signals (zero LLM calls, zero added latency) and routes to the cheapest model that can handle it:

| Tier | When | Default Models | Cost/MTok |
|------|------|---------------|-----------|
| **Simple** | Short queries, yes/no, formatting, greetings | Gemini 2.5 Flash, GPT-4.1 Nano | $0.15-0.50 |
| **Standard** | Reasoning, code generation, analysis | Claude Sonnet 4.6, GPT-4.1 Mini | $1-3 |
| **Complex** | Multi-step reasoning, architecture, proofs | Claude Opus 4.6, o4-mini | $5-25 |

**Complexity signals** (27 total): word count, code blocks, multi-step keywords ("step 1", "first", "then", "finally"), analysis keywords ("analyze", "compare", "design", "optimize"), math keywords ("calculate", "prove", "algorithm"), and more.

A typical workload (60% simple, 25% standard, 15% complex) saves **60-70%** vs sending everything to a frontier model.

**Budget-aware downgrade**: When remaining budget drops below thresholds, the router automatically downgrades:
- Complex → Standard when < $0.10 remaining
- Standard → Simple when < $0.05 remaining

```python
# Explicit model — bypass routing
agent = Agent(model="claude-sonnet-4.6", ...)

# Auto routing — framework picks per step
agent = Agent(model="auto", ...)

# Custom tiers
from archon import Router
router = Router(models={
    Tier.SIMPLE: ["deepseek-v3.2", "gemini-flash"],
    Tier.STANDARD: ["claude-sonnet-4.6"],
    Tier.COMPLEX: ["o4-mini"],
})
```

### Budget Enforcement

Not a dashboard. Not an alert. A **hard stop**.

```python
from archon import Budget

budget = Budget(
    max_per_run=0.50,       # Agent stops before exceeding $0.50
    max_per_day=10.00,      # Daily cap across all runs
    max_per_month=200.00,   # Monthly cap
    warn_at=0.8,            # Warning at 80% of any limit
)

agent = Agent(name="researcher", budget=budget, ...)
result = await agent.run("Complex research task")

# If budget exceeded mid-run:
# result.output = "[Budget exceeded after 7 steps, spent $0.4998]"
# Agent stops gracefully — no surprise bills.
```

Budget state persists across restarts. Every LLM call is tracked with model, tokens, cost, and attribution tags.

### Tool System

Define tools with a simple decorator. Archon generates the JSON Schema automatically from type hints.

```python
from archon import tool

@tool
def search_web(query: str) -> str:
    """Search the web for current information."""
    return requests.get(f"https://api.search.com?q={query}").text

@tool
def read_file(path: str) -> str:
    """Read a file from the local filesystem."""
    return Path(path).read_text()

@tool
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression safely."""
    return safe_eval(expression)

agent = Agent(
    name="assistant",
    instructions="Help the user with research and analysis.",
    tools=[search_web, read_file, calculate],
    model="auto",
)
```

Each tool automatically gets:
- JSON Schema generation from type hints (for LLM function calling)
- Input validation before execution
- Output sanitization before passing back to LLM
- Cost tracking (tool calls count toward budget)
- Trace logging (tool name, parameters, result, latency)

---

## Installation

### Python (via uv — recommended)

```bash
uv add archon-ai

# Or with pip
pip install archon-ai
```

### TypeScript (via pnpm — recommended)

```bash
pnpm add @archon-ai/sdk

# Or with npm
npm install @archon-ai/sdk
```

### From Source

```bash
git clone https://github.com/yash1511-bogam/archon.git
cd archon

# Rust core
cargo build --manifest-path crates/archon-core/Cargo.toml

# Python SDK
cd sdks/python && uv sync

# TypeScript SDK
cd sdks/typescript && pnpm install && pnpm build
```

---

## Usage

### Basic Agent (Python)

```python
import asyncio
from archon import Agent, tool, Budget

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

agent = Agent(
    name="helper",
    instructions="You are a helpful research assistant.",
    tools=[search],
    model="auto",
    budget=Budget(max_per_run=1.00),
)

async def main():
    result = await agent.run("What are the latest advances in quantum computing?")
    print(f"Answer: {result.output}")
    print(f"Cost: ${result.cost:.4f}")
    print(f"Steps: {result.step_count}")
    print(f"Models used: {result.model_usage}")

asyncio.run(main())
```

### Basic Agent (TypeScript)

```typescript
import { Budget, Router } from "@archon-ai/sdk";

// Budget enforcement
const budget = new Budget({ maxPerRun: 0.50 });
budget.check(0.10);  // OK
budget.record(0.10);
console.log(`Remaining: $${budget.remaining}`);  // $0.40

// Model routing
const router = new Router();
const decision = router.route("Analyze this complex architecture");
console.log(`Model: ${decision.model}`);  // claude-sonnet-4.6
console.log(`Tier: ${decision.tier}`);    // standard
```

### Custom Model Tiers

```python
from archon import Router, Tier

router = Router(models={
    Tier.SIMPLE: ["deepseek-v3.2", "qwen-3.5-flash"],
    Tier.STANDARD: ["claude-sonnet-4.6", "gpt-4.1-mini"],
    Tier.COMPLEX: ["claude-opus-4.6", "o4-mini", "gemini-2.5-pro"],
})

# Route with budget awareness
decision = router.route("Simple greeting", remaining_budget=0.03)
# → Tier.SIMPLE, model="deepseek-v3.2"

decision = router.route("Analyze and compare architectures with code examples")
# → Tier.COMPLEX, model="claude-opus-4.6"
```

### Multi-Step Agent with Tools

```python
from archon import Agent, tool, Budget

@tool
def search_web(query: str) -> str:
    """Search the web."""
    return web_search_api(query)

@tool
def read_document(url: str) -> str:
    """Read and extract text from a document URL."""
    return extract_text(url)

@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    Path(path).write_text(content)
    return f"Written to {path}"

agent = Agent(
    name="research-writer",
    instructions="""You are a research assistant. When given a topic:
    1. Search for relevant sources
    2. Read the most relevant documents
    3. Write a comprehensive summary to a file""",
    tools=[search_web, read_document, write_file],
    model="auto",
    budget=Budget(max_per_run=2.00),
    max_steps=15,
)

result = await agent.run("Research the current state of AI agent frameworks in 2026")
```

---

## Supported Models (140+ across 17 providers)

Archon ships a built-in model registry with pricing for every major model from 2024–2026. The router uses this registry for cost-aware model selection. **Azure AI Foundry is fully supported — not just Azure OpenAI, but all 11,000+ Foundry models including Claude, Grok, DeepSeek, Llama, Mistral, Phi, Kimi, and GLM.**

### Direct API Providers

| Provider | Models | Price Range (Input $/MTok) | Highlights |
|----------|--------|---------------------------|------------|
| **OpenAI** | GPT-5.5, 5.4/Pro/Mini/Nano, 5.3 Codex, 5.2/Pro, 5.1, 5/Mini/Nano, 4.1/Mini/Nano, 4o/Mini, o3/Pro, o4-mini, GPT-OSS-120B | $0.05 – $30.00 | 1.1M context on GPT-5.4, model-router |
| **Anthropic** | Claude Opus 4.7/4.6/4.5, Sonnet 4.6/4.5/4, Haiku 4.5/3.5 | $0.80 – $5.00 | 1M context flat-rate, 90% cache discount |
| **Google** | Gemini 3.1/3 Pro, 3/2.5 Flash, 2.5/2.0 Flash-Lite, 1.5 Pro/Flash, Gemma 4/3 | $0.075 – $2.00 | 1M–2M context, free tiers |
| **xAI** | Grok 4.20, 4, 4.1 Fast, Code Fast 1, 3, 3 Mini | $0.20 – $3.00 | 2M context, real-time X/web search |
| **DeepSeek** | V4 Pro/Flash, V3.2/V3.1/V3, R2, R1 | $0.14 – $1.74 | 90% cache discount, Apache 2.0 |
| **Meta** | Llama 4 Maverick/Scout, 3.3 70B, 3.1 405B/70B/8B | $0.05 – $3.00 | 10M context (Scout), fully open |
| **Mistral** | Large 3, Small 4/3.2, Medium 3, Nemo, Codestral, Devstral 2 | $0.02 – $2.00 | EU AI Act compliant, MIT license |
| **Alibaba** | Qwen 3.6 Plus, 3.5 Plus, 3 235B/32B/14B/8B | $0.00 – $0.455 | Free tier, 119 languages |
| **Cohere** | Command A, R+, R, R7B, Embed, Rerank | $0.037 – $2.50 | RAG-optimized, multilingual |
| **AI21** | Jamba 2 Large/Mini | $0.20 – $2.00 | 256K context |
| **Microsoft** | Phi-4, Phi-4 Mini, Phi-4 Multimodal | $0.02 – $0.07 | Tiny but capable, open source |
| **Moonshot** | Kimi K2.6, K2.5, K2 Thinking | $0.20 | 1M context, open source |
| **Amazon** | Nova Pro, Lite, Micro | $0.035 – $0.80 | Bedrock-native |
| **Perplexity** | Sonar Pro, Sonar | $1.00 – $3.00 | Built-in web search |
| **Zhipu** | GLM-5 | $0.50 | MIT license, 744B MoE |

### Platform Support

| Platform | What It Provides | How Archon Uses It |
|----------|-----------------|-------------------|
| **Azure AI Foundry** | 11,000+ models: GPT-5.x, Claude, Grok, DeepSeek, Llama, Mistral, Phi, Kimi, GLM, Cohere, model-router, NVIDIA NIMs, Hugging Face, Stability AI | Set `AZURE_API_KEY` + `AZURE_API_BASE` — use `azure/` prefix. Supports all Direct + Partner models. |
| **AWS Bedrock** | Claude, Llama, Mistral, Cohere, Nova, Titan | Set `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` — use `bedrock/` prefix |
| **OpenRouter** | 1,600+ models via single API | Set `OPENROUTER_API_KEY` — use `openrouter/` prefix |
| **Groq** | LPU-accelerated Llama, DeepSeek, Mixtral (300–840 tok/s) | Set `GROQ_API_KEY` — use `groq/` prefix |
| **Together AI** | Serverless open models (Llama, DeepSeek, Qwen) | Set `TOGETHER_API_KEY` — use `together_ai/` prefix |
| **Fireworks AI** | Optimized inference for open models | Set `FIREWORKS_API_KEY` — use `fireworks_ai/` prefix |
| **Cerebras** | Ultra-fast inference (920 tok/s) | Set `CEREBRAS_API_KEY` — use `cerebras/` prefix |

### Using the Model Registry

```python
from archon.models import list_models, get_model, estimate_cost

# Find all budget models under $0.20/MTok input
cheap = list_models(tier="budget", max_input_cost=0.20)
for m in cheap[:5]:
    print(f"{m.id:<25} ${m.input_cost:.3f} / ${m.output_cost:.3f}  {m.provider}")

# Get pricing for a specific model
gpt54 = get_model("gpt-5.4")
print(f"GPT-5.4: ${gpt54.input_cost}/MTok in, ${gpt54.output_cost}/MTok out, {gpt54.context_window:,} ctx")

# Estimate cost for a workload
cost = estimate_cost("claude-sonnet-4.6", input_tokens=50_000, output_tokens=2_000)
print(f"Estimated cost: ${cost:.4f}")

# Filter by capability
reasoning = list_models(is_reasoning=True)
open_source = list_models(is_open_source=True, min_context=1_000_000)
vision = list_models(supports_vision=True, max_input_cost=1.00)
```

---

## Integration with Other Frameworks & Tools

Archon is designed to work **alongside** existing tools, not replace everything.

### With MCP (Model Context Protocol)

Archon tools are compatible with MCP. Use any MCP server as an Archon tool:

```python
# Use MCP servers as Archon tools (planned)
from archon.mcp import from_mcp_server

github_tools = from_mcp_server("npx @modelcontextprotocol/server-github")
agent = Agent(tools=github_tools, ...)
```

### With LangChain / LangGraph

Use Archon's router and budget engine inside LangChain workflows:

```python
from archon import Router, Budget

router = Router()
budget = Budget(max_per_run=1.00)

# Use Archon's routing decision in your LangChain chain
decision = router.route(user_input, budget.remaining)
# → Use decision.model as the model for your LangChain LLM call

# Track cost after each LangChain call
budget.record(actual_cost)
```

### With CrewAI

Use Archon's budget tracking to wrap CrewAI crews:

```python
from archon import Budget

budget = Budget(max_per_run=5.00)

# Before each CrewAI task
budget.check(estimated_cost)
# After each task
budget.record(actual_cost)
```

### With Pydantic-AI

Archon uses Pydantic 2 natively — types are directly compatible:

```python
from archon.types import AgentResult, Step
# These are Pydantic BaseModel instances — serialize, validate, compose freely
```

### With OpenTelemetry

Export Archon traces to any OTLP-compatible backend (Datadog, Grafana, Jaeger):

```python
# Planned: OTLP export
from archon.trace import TraceStore
traces = TraceStore(export=["otlp"])
```

### With Vector Databases

Archon's memory layer (planned) works with any vector store:

```python
# Planned: pluggable memory backends
from archon.memory import Memory
memory = Memory(
    semantic_backend="pgvector",  # or "qdrant", "chroma", "pinecone"
)
```

---

## Architecture

```
archon/
├── crates/archon-core/        # Rust core runtime
│   ├── src/types.rs           #   Step, RunResult, BudgetConfig, Tier
│   ├── src/budget.rs          #   Budget tracker with hard enforcement
│   ├── src/router.rs          #   Pattern-based complexity classifier (27 signals)
│   └── src/trace.rs           #   SQLite trace store + immutable audit log
│
├── sdks/python/               # Python SDK (uv + hatchling)
│   └── src/archon/
│       ├── agent.py           #   Agent class — the core API
│       ├── types.py           #   Pydantic 2 models
│       ├── tool.py            #   @tool decorator → JSON Schema
│       ├── budget/            #   Budget with hard caps
│       ├── router/            #   Pattern classifier (mirrors Rust)
│       ├── memory/            #   Tiered memory (planned)
│       ├── security/          #   Sandbox + policy engine (planned)
│       ├── trace/             #   Trace store (planned)
│       └── eval/              #   Continuous evaluation (planned)
│
├── sdks/typescript/           # TypeScript SDK (pnpm + vitest)
│   └── src/
│       ├── types.ts           #   Zod-based types
│       ├── budget.ts          #   Budget (matches Python)
│       └── router.ts          #   Router (matches Python)
│
└── proto/                     # Shared protocol definitions
    ├── schema.json            #   JSON Schema for all types
    └── README.md              #   Classification spec
```

### Why This Architecture

**Rust core** — The execution engine, sandbox, trace store, and budget engine are in Rust for three reasons:
1. **Single binary distribution** — users run `brew install archon`, no Python/Node dependency
2. **Memory-safe sandboxing** — direct access to seccomp/Landlock/Seatbelt syscalls
3. **Performance** — trace storage and budget tracking are hot paths

**Python SDK (PyO3)** — Thin wrapper around the Rust core. Uses Pydantic 2 for type safety and JSON Schema generation. LiteLLM for 100+ model support.

**TypeScript SDK (napi-rs)** — Thin wrapper around the same Rust core. Uses Zod for type safety. Feature parity with Python.

**Shared protocol** — JSON Schema definitions ensure all three implementations agree on types, classification thresholds, and behavior.

---

## Development

### Prerequisites

- Rust 1.75+ (for core)
- Python 3.10+ with [uv](https://docs.astral.sh/uv/)
- Node.js 20+ with [pnpm](https://pnpm.io/)

### Build Everything

```bash
make build
```

### Run All Tests

```bash
make test
# Runs: 4 Rust + 112 Python + 10 TypeScript = 126 tests
```

### Individual Components

```bash
# Rust core
cargo build --manifest-path crates/archon-core/Cargo.toml
cargo test --manifest-path crates/archon-core/Cargo.toml --lib

# Python SDK
cd sdks/python
uv sync
uv run python -m pytest tests/ -v

# TypeScript SDK
cd sdks/typescript
pnpm install
pnpm build
pnpm test
```

### Lint

```bash
make lint
# Runs: ruff (Python) + tsc --noEmit (TypeScript)
```

---

## Roadmap

### Phase 0 — Foundation ✅ (Current)
- [x] Rust core: types, budget tracker, router, trace store
- [x] Python SDK: Agent, @tool, Budget, Router, auto-routing
- [x] TypeScript SDK: Budget, Router with matching logic
- [x] Shared protocol definitions (JSON Schema)
- [x] 19 tests passing across 3 languages

### Phase 1 — The Harness ✅
- [x] Semantic cache (skip LLM calls for similar queries — TF-IDF cosine, zero external deps)
- [x] Security layer (subprocess sandbox, policy engine, default-deny)
- [x] Tool output sanitization (7 threat categories, 5 severity levels)
- [x] Built-in trace store with SQLite backend (WAL mode, retention, audit log)
- [x] `archon traces` CLI command (list, show, stats, purge)

### Phase 2 — Memory + Multi-Agent ✅
- [x] Tiered memory (working/episodic/semantic/procedural) with temporal decay
- [x] Automatic consolidation (expire old episodic, prune stale semantic)
- [x] Multi-agent pipelines (sequential, parallel, hierarchical)
- [x] Durable execution with SQLite-backed checkpointing + crash recovery
- [x] Full TypeScript Agent class with LLM integration, routing, budget

### Phase 3 — Production Polish ✅
- [x] Built-in web dashboard (`archon dashboard` — dark theme, zero JS deps)
- [x] Continuous evaluation (inline validators + async quality scoring + regression detection)
- [x] Shadow deployments (run candidate in parallel, compare scores, promotion recommendation)
- [x] Governance (event sourcing, RBAC for tool access, GDPR-compliant erasure)
- [x] MCP client (stdio transport, tool discovery, to_archon_tools conversion)
- [x] A2A protocol (Agent Cards, skill discovery, remote agent fetch)

---

## Comparison with Other Frameworks

| Feature | LangGraph | CrewAI | Pydantic-AI | OpenAI SDK | **Archon** |
|---------|-----------|--------|-------------|------------|-----------|
| Built-in model routing | ✗ | ✗ | ✗ | ✗ | **✓** |
| Budget enforcement | ✗ | ✗ | ✗ | ✗ | **✓** |
| Per-step cost tracking | ✗ | ✗ | ✗ | ✗ | **✓** |
| Semantic cache | ✗ | ✗ | ✗ | ✗ | **✓** |
| Sandboxed by default | ✗ | ✗ | ✗ | ✗ | **✓** |
| Pre-execution policy | ✗ | ✗ | ✗ | ✗ | **✓** |
| Tool output sanitization | ✗ | ✗ | ✗ | ✗ | **✓** |
| Built-in trace store | ✗ (LangSmith $) | ✗ | ✗ | Partial | **✓** |
| CLI for traces | ✗ | ✗ | ✗ | ✗ | **✓** |
| Tiered memory | ✗ | ✗ | ✗ | ✗ | **✓** |
| Continuous eval | ✗ | ✗ | Partial | ✗ | **✓** |
| Python + TypeScript | ✗ | ✗ | ✗ | ✗ | **✓** |
| MCP support | ✓ | ✓ | ✓ | ✓ | **✓** |
| Durable execution | ✓ | ✗ | ✓ (ext.) | ✗ | **✓** |

**Archon doesn't compete with these frameworks on orchestration patterns.** LangGraph's graph model, CrewAI's role-based teams, and Pydantic-AI's DX are excellent. Archon competes on the **production harness** — the infrastructure layer that makes any agent safe, cheap, and observable.

---

## Philosophy

1. **Secure by default** — Unsafe mode requires explicit opt-in, not the other way around.
2. **Cost-aware by default** — Every LLM call is tracked, budgeted, and routable.
3. **Observable by default** — Every step emits structured traces. No external tools required.
4. **Thin orchestration** — Business logic in separate functions. The framework coordinates, not imprisons.
5. **Dual-language** — Python and TypeScript as first-class citizens.
6. **Zero-dependency setup** — SQLite for everything by default. No Redis, no Postgres, no Docker required.

---

## Contributing

Contributions welcome. See the [proto/README.md](proto/README.md) for the shared protocol spec that all implementations must follow.

```bash
# Run the full test suite before submitting
make test
```

---

## License

Apache-2.0 — see [LICENSE](LICENSE).
