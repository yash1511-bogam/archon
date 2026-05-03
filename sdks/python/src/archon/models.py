"""Comprehensive model registry — every provider, every model, every price.

Covers 2024-2026 models across all major providers and platforms:
  - OpenAI (GPT-5.x, GPT-4.x, o-series, GPT-OSS)
  - Anthropic (Claude Opus/Sonnet/Haiku 3.5-4.7)
  - Google (Gemini 1.5-3.1, Gemma)
  - xAI (Grok 3-4.20)
  - DeepSeek (V3-V4, R1-R2)
  - Meta (Llama 3.x-4.x)
  - Mistral (Large 3, Small 4, Magistral, Devstral, Nemo)
  - Alibaba (Qwen 2.5-3.6)
  - Cohere (Command R/R+/A, Embed, Rerank)
  - AI21 (Jamba 2)
  - Microsoft (Phi-4.x, MAI-DS-R1)
  - Moonshot (Kimi K2-K2.6)
  - Perplexity (Sonar, Sonar Pro)

Platform support:
  - Azure AI Foundry (11,000+ models, model-router)
  - AWS Bedrock (Nova, Claude, Llama, Mistral, Cohere)
  - OpenRouter (1,600+ models)
  - Groq (LPU-accelerated open models)
  - Together AI (serverless open models)
  - Fireworks AI (optimized inference)
  - Cerebras (ultra-fast inference)

Pricing is per 1M tokens (USD), updated April 2026.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    """Pricing and metadata for a single model.

    Attributes:
        id: Model identifier used in API calls (e.g., "gpt-5.4").
        provider: Provider name (e.g., "openai", "anthropic").
        input_cost: USD per 1M input tokens.
        output_cost: USD per 1M output tokens.
        context_window: Maximum context length in tokens.
        supports_tools: Whether the model supports function/tool calling.
        supports_vision: Whether the model accepts image inputs.
        is_reasoning: Whether this is a reasoning/thinking model.
        is_open_source: Whether model weights are publicly available.
        tier: Suggested routing tier ("budget", "mid", "flagship", "reasoning", "ultra").
    """

    id: str
    provider: str
    input_cost: float
    output_cost: float
    context_window: int = 128_000
    supports_tools: bool = True
    supports_vision: bool = False
    is_reasoning: bool = False
    is_open_source: bool = False
    tier: str = "mid"


# ══════════════════════════════════════════════════════
# Model Registry — All Providers (April 2026)
# ══════════════════════════════════════════════════════

MODELS: dict[str, ModelInfo] = {}


def _register(*models: ModelInfo) -> None:
    for m in models:
        MODELS[m.id] = m


# ── OpenAI ─────────────────────────────────────────────

_register(
    # GPT-5.x series
    ModelInfo("gpt-5.5",            "openai",   3.00,   18.00,  1_100_000, supports_vision=True, tier="flagship"),
    ModelInfo("gpt-5.4",            "openai",   2.50,   15.00,  1_100_000, supports_vision=True, tier="flagship"),
    ModelInfo("gpt-5.4-pro",        "openai",  30.00,  180.00,  1_100_000, supports_vision=True, is_reasoning=True, tier="ultra"),
    ModelInfo("gpt-5.4-mini",       "openai",   0.75,    4.50,  1_100_000, supports_vision=True, tier="mid"),
    ModelInfo("gpt-5.4-nano",       "openai",   0.20,    1.25,  1_100_000, tier="budget"),
    ModelInfo("gpt-5.3-codex",      "openai",   1.75,   14.00,    400_000, tier="mid"),
    ModelInfo("gpt-5.2",            "openai",   1.75,   14.00,    400_000, supports_vision=True, tier="mid"),
    ModelInfo("gpt-5.2-pro",        "openai",  21.00,  168.00,    400_000, is_reasoning=True, tier="ultra"),
    ModelInfo("gpt-5.1",            "openai",   1.25,   10.00,    400_000, supports_vision=True, tier="mid"),
    ModelInfo("gpt-5.1-codex",      "openai",   1.25,   10.00,    400_000, tier="mid"),
    ModelInfo("gpt-5",              "openai",   1.25,   10.00,    400_000, supports_vision=True, tier="mid"),
    ModelInfo("gpt-5-mini",         "openai",   0.25,    2.00,    128_000, supports_vision=True, tier="budget"),
    ModelInfo("gpt-5-nano",         "openai",   0.05,    0.40,    400_000, tier="budget"),
    # GPT-4.x series
    ModelInfo("gpt-4.1",            "openai",   2.00,    8.00,  1_000_000, supports_vision=True, tier="mid"),
    ModelInfo("gpt-4.1-mini",       "openai",   0.40,    1.60,  1_000_000, supports_vision=True, tier="budget"),
    ModelInfo("gpt-4.1-nano",       "openai",   0.10,    0.40,  1_000_000, tier="budget"),
    ModelInfo("gpt-4o",             "openai",   2.50,   10.00,    128_000, supports_vision=True, tier="mid"),
    ModelInfo("gpt-4o-mini",        "openai",   0.15,    0.60,    128_000, supports_vision=True, tier="budget"),
    # o-series (reasoning)
    ModelInfo("o3",                 "openai",   2.00,    8.00,    200_000, is_reasoning=True, tier="reasoning"),
    ModelInfo("o3-pro",             "openai",  20.00,   80.00,    200_000, is_reasoning=True, tier="ultra"),
    ModelInfo("o4-mini",            "openai",   1.10,    4.40,    200_000, is_reasoning=True, tier="reasoning"),
    ModelInfo("o3-mini",            "openai",   1.10,    4.40,    128_000, is_reasoning=True, tier="reasoning"),
    # Open-weight
    ModelInfo("gpt-oss-120b",       "openai",   0.15,    0.60,    128_000, is_open_source=True, tier="budget"),
)

# ── Anthropic ──────────────────────────────────────────

_register(
    ModelInfo("claude-opus-4.7",    "anthropic",  5.00,  25.00,  1_000_000, supports_vision=True, tier="flagship"),
    ModelInfo("claude-opus-4.6",    "anthropic",  5.00,  25.00,  1_000_000, supports_vision=True, tier="flagship"),
    ModelInfo("claude-opus-4.5",    "anthropic",  5.00,  25.00,  1_000_000, supports_vision=True, tier="flagship"),
    ModelInfo("claude-sonnet-4.6",  "anthropic",  3.00,  15.00,  1_000_000, supports_vision=True, tier="mid"),
    ModelInfo("claude-sonnet-4.5",  "anthropic",  3.00,  15.00,  1_000_000, supports_vision=True, tier="mid"),
    ModelInfo("claude-sonnet-4",    "anthropic",  3.00,  15.00,    200_000, supports_vision=True, tier="mid"),
    ModelInfo("claude-haiku-4.5",   "anthropic",  1.00,   5.00,    200_000, supports_vision=True, tier="budget"),
    ModelInfo("claude-haiku-3.5",   "anthropic",  0.80,   4.00,    200_000, supports_vision=True, tier="budget"),
)

# ── Google ─────────────────────────────────────────────

_register(
    ModelInfo("gemini-3.1-pro",     "google",   2.00,  12.00,  1_000_000, supports_vision=True, tier="flagship"),
    ModelInfo("gemini-3-pro",       "google",   2.00,  12.00,  1_000_000, supports_vision=True, tier="flagship"),
    ModelInfo("gemini-3-flash",     "google",   0.50,   3.00,  1_000_000, supports_vision=True, tier="mid"),
    ModelInfo("gemini-3.1-flash-lite", "google", 0.25,  1.50,  1_000_000, supports_vision=True, tier="budget"),
    ModelInfo("gemini-2.5-pro",     "google",   1.25,  10.00,  1_000_000, supports_vision=True, tier="mid"),
    ModelInfo("gemini-2.5-flash",   "google",   0.30,   2.50,  1_000_000, supports_vision=True, tier="budget"),
    ModelInfo("gemini-2.5-flash-lite", "google", 0.10,  0.40,  1_000_000, supports_vision=True, tier="budget"),
    ModelInfo("gemini-2.0-flash",   "google",   0.10,   0.40,  1_000_000, supports_vision=True, tier="budget"),
    ModelInfo("gemini-2.0-flash-lite", "google", 0.075, 0.30,  1_000_000, supports_vision=True, tier="budget"),
    ModelInfo("gemini-1.5-pro",     "google",   1.25,   5.00,  2_000_000, supports_vision=True, tier="mid"),
    ModelInfo("gemini-1.5-flash",   "google",   0.075,  0.30,  1_000_000, supports_vision=True, tier="budget"),
)

# ── xAI (Grok) ────────────────────────────────────────

_register(
    ModelInfo("grok-4.20",          "xai",      2.00,   6.00,  2_000_000, supports_vision=True, is_reasoning=True, tier="flagship"),
    ModelInfo("grok-4",             "xai",      3.00,  15.00,    256_000, is_reasoning=True, tier="flagship"),
    ModelInfo("grok-4.1-fast",      "xai",      0.20,   0.50,  2_000_000, tier="budget"),
    ModelInfo("grok-code-fast-1",   "xai",      0.20,   1.50,    256_000, tier="budget"),
    ModelInfo("grok-3",             "xai",      3.00,  15.00,    131_000, tier="flagship"),
    ModelInfo("grok-3-mini",        "xai",      0.30,   0.50,    131_000, tier="budget"),
)

# ── DeepSeek ───────────────────────────────────────────

_register(
    ModelInfo("deepseek-v4-pro",    "deepseek", 1.74,   3.48,  1_000_000, is_open_source=True, tier="mid"),
    ModelInfo("deepseek-v4-flash",  "deepseek", 0.14,   0.28,  1_000_000, is_open_source=True, tier="budget"),
    ModelInfo("deepseek-v3.2",      "deepseek", 0.28,   0.42,    128_000, is_open_source=True, tier="budget"),
    ModelInfo("deepseek-v3.1",      "deepseek", 0.27,   1.10,    128_000, is_open_source=True, tier="budget"),
    ModelInfo("deepseek-v3",        "deepseek", 0.27,   1.10,     64_000, is_open_source=True, tier="budget"),
    ModelInfo("deepseek-r2",        "deepseek", 0.45,   2.15,    128_000, is_open_source=True, is_reasoning=True, tier="reasoning"),
    ModelInfo("deepseek-r1",        "deepseek", 0.55,   2.19,    128_000, is_open_source=True, is_reasoning=True, tier="reasoning"),
)

# ── Meta (Llama) ───────────────────────────────────────

_register(
    ModelInfo("llama-4-maverick",   "meta",     0.27,   0.85,  1_000_000, is_open_source=True, tier="budget"),
    ModelInfo("llama-4-scout",      "meta",     0.08,   0.30, 10_000_000, is_open_source=True, tier="budget"),
    ModelInfo("llama-3.3-70b",      "meta",     0.18,   0.18,    131_000, is_open_source=True, tier="budget"),
    ModelInfo("llama-3.1-405b",     "meta",     3.00,   3.00,    128_000, is_open_source=True, tier="mid"),
    ModelInfo("llama-3.1-70b",      "meta",     0.18,   0.18,    128_000, is_open_source=True, tier="budget"),
    ModelInfo("llama-3.1-8b",       "meta",     0.05,   0.08,    128_000, is_open_source=True, tier="budget"),
)

# ── Mistral ────────────────────────────────────────────

_register(
    ModelInfo("mistral-large-3",    "mistral",  2.00,   6.00,    128_000, is_open_source=True, tier="mid"),
    ModelInfo("mistral-small-4",    "mistral",  0.10,   0.30,    128_000, is_open_source=True, tier="budget"),
    ModelInfo("mistral-small-3.2",  "mistral",  0.07,   0.20,    128_000, is_open_source=True, tier="budget"),
    ModelInfo("mistral-medium-3",   "mistral",  0.40,   2.00,    128_000, tier="mid"),
    ModelInfo("mistral-nemo",       "mistral",  0.02,   0.04,    128_000, is_open_source=True, tier="budget"),
    ModelInfo("codestral",          "mistral",  0.30,   0.90,     32_000, is_open_source=True, tier="budget"),
    ModelInfo("devstral-2",         "mistral",  0.30,   0.90,    256_000, is_open_source=True, tier="budget"),
)

# ── Alibaba (Qwen) ────────────────────────────────────

_register(
    ModelInfo("qwen-3.6-plus",      "alibaba",  0.00,   0.00,  1_000_000, is_open_source=False, tier="mid"),
    ModelInfo("qwen-3.5-plus",      "alibaba",  0.065,  0.26,  1_000_000, is_open_source=True, tier="budget"),
    ModelInfo("qwen-3-235b",        "alibaba",  0.455,  1.82,    128_000, is_open_source=True, tier="mid"),
    ModelInfo("qwen-3-32b",         "alibaba",  0.29,   0.59,    131_000, is_open_source=True, tier="budget"),
    ModelInfo("qwen-3-14b",         "alibaba",  0.10,   0.20,    128_000, is_open_source=True, tier="budget"),
    ModelInfo("qwen-3-8b",          "alibaba",  0.05,   0.10,    128_000, is_open_source=True, tier="budget"),
)

# ── Cohere ─────────────────────────────────────────────

_register(
    ModelInfo("command-a",          "cohere",   2.50,  10.00,    256_000, tier="mid"),
    ModelInfo("command-r-plus",     "cohere",   2.50,  10.00,    128_000, tier="mid"),
    ModelInfo("command-r",          "cohere",   0.50,   1.50,    128_000, tier="budget"),
    ModelInfo("command-r7b",        "cohere",   0.037,  0.15,    128_000, tier="budget"),
)

# ── AI21 ───────────────────────────────────────────────

_register(
    ModelInfo("jamba-2-large",      "ai21",     2.00,   8.00,    256_000, tier="mid"),
    ModelInfo("jamba-2-mini",       "ai21",     0.20,   0.40,    256_000, tier="budget"),
)

# ── Microsoft (Phi) ────────────────────────────────────

_register(
    ModelInfo("phi-4",              "microsoft", 0.07,  0.14,    128_000, is_open_source=True, tier="budget"),
    ModelInfo("phi-4-mini",         "microsoft", 0.02,  0.04,    128_000, is_open_source=True, tier="budget"),
    ModelInfo("phi-4-multimodal",   "microsoft", 0.07,  0.14,    128_000, is_open_source=True, supports_vision=True, tier="budget"),
)

# ── Moonshot (Kimi) ────────────────────────────────────

_register(
    ModelInfo("kimi-k2.6",         "moonshot",  0.20,   0.80,  1_000_000, is_open_source=True, tier="budget"),
    ModelInfo("kimi-k2.5",         "moonshot",  0.20,   0.80,  1_000_000, is_open_source=True, supports_vision=True, tier="budget"),
    ModelInfo("kimi-k2-thinking",  "moonshot",  0.20,   0.80,  1_000_000, is_open_source=True, is_reasoning=True, tier="reasoning"),
)

# ── Amazon (Nova) ──────────────────────────────────────

_register(
    ModelInfo("nova-pro",           "amazon",   0.80,   3.20,    300_000, supports_vision=True, tier="mid"),
    ModelInfo("nova-lite",          "amazon",   0.06,   0.24,    300_000, supports_vision=True, tier="budget"),
    ModelInfo("nova-micro",         "amazon",   0.035,  0.14,    128_000, tier="budget"),
)

# ── Perplexity ─────────────────────────────────────────

_register(
    ModelInfo("sonar-pro",          "perplexity", 3.00, 15.00,   200_000, tier="mid"),
    ModelInfo("sonar",              "perplexity", 1.00,  1.00,   200_000, tier="budget"),
)

# ── Google (Gemma — open source) ───────────────────────

_register(
    ModelInfo("gemma-4-31b",        "google",   0.10,   0.20,    128_000, is_open_source=True, tier="budget"),
    ModelInfo("gemma-3-27b",        "google",   0.10,   0.20,    128_000, is_open_source=True, tier="budget"),
)

# ── GLM (Zhipu) ───────────────────────────────────────

_register(
    ModelInfo("glm-5",              "zhipu",    0.50,   1.00,    128_000, is_open_source=True, tier="mid"),
)


# ══════════════════════════════════════════════════════
# Azure AI Foundry — Direct from Azure models
# (These are the same models but accessed via Azure endpoints.
#  LiteLLM routes them with the "azure/" prefix.)
# ══════════════════════════════════════════════════════

_register(
    # Azure-hosted OpenAI (same pricing, Azure SLA)
    ModelInfo("azure/gpt-5.5",          "azure",    3.00,   18.00,  1_100_000, supports_vision=True, tier="flagship"),
    ModelInfo("azure/gpt-5.4",          "azure",    2.50,   15.00,  1_100_000, supports_vision=True, tier="flagship"),
    ModelInfo("azure/gpt-5.4-pro",      "azure",   30.00,  180.00,  1_100_000, is_reasoning=True, tier="ultra"),
    ModelInfo("azure/gpt-5.4-mini",     "azure",    0.75,    4.50,  1_100_000, supports_vision=True, tier="mid"),
    ModelInfo("azure/gpt-5.4-nano",     "azure",    0.20,    1.25,  1_100_000, tier="budget"),
    ModelInfo("azure/gpt-5.3-codex",    "azure",    1.75,   14.00,    400_000, tier="mid"),
    ModelInfo("azure/gpt-5.2",          "azure",    1.75,   14.00,    400_000, tier="mid"),
    ModelInfo("azure/gpt-5.1",          "azure",    1.25,   10.00,    400_000, tier="mid"),
    ModelInfo("azure/gpt-5",            "azure",    1.25,   10.00,    400_000, tier="mid"),
    ModelInfo("azure/gpt-5-mini",       "azure",    0.25,    2.00,    128_000, tier="budget"),
    ModelInfo("azure/gpt-5-nano",       "azure",    0.05,    0.40,    400_000, tier="budget"),
    ModelInfo("azure/gpt-4.1",          "azure",    2.00,    8.00,  1_000_000, supports_vision=True, tier="mid"),
    ModelInfo("azure/gpt-4.1-mini",     "azure",    0.40,    1.60,  1_000_000, supports_vision=True, tier="budget"),
    ModelInfo("azure/gpt-4.1-nano",     "azure",    0.10,    0.40,  1_000_000, tier="budget"),
    ModelInfo("azure/gpt-4o",           "azure",    2.50,   10.00,    128_000, supports_vision=True, tier="mid"),
    ModelInfo("azure/gpt-4o-mini",      "azure",    0.15,    0.60,    128_000, supports_vision=True, tier="budget"),
    ModelInfo("azure/o3",               "azure",    2.00,    8.00,    200_000, is_reasoning=True, tier="reasoning"),
    ModelInfo("azure/o4-mini",          "azure",    1.10,    4.40,    200_000, is_reasoning=True, tier="reasoning"),
    ModelInfo("azure/o3-mini",          "azure",    1.10,    4.40,    128_000, is_reasoning=True, tier="reasoning"),
    ModelInfo("azure/gpt-oss-120b",     "azure",    0.15,    0.60,    128_000, is_open_source=True, tier="budget"),
    ModelInfo("azure/model-router",     "azure",    0.00,    0.00,    128_000, tier="budget"),  # Routes to best model

    # Azure-hosted Anthropic Claude (via Foundry partner deployment)
    ModelInfo("azure/claude-opus-4.6",  "azure",    5.00,   25.00,  1_000_000, supports_vision=True, tier="flagship"),
    ModelInfo("azure/claude-sonnet-4.6","azure",    3.00,   15.00,  1_000_000, supports_vision=True, tier="mid"),
    ModelInfo("azure/claude-sonnet-4.5","azure",    3.00,   15.00,  1_000_000, supports_vision=True, tier="mid"),
    ModelInfo("azure/claude-opus-4.5",  "azure",    5.00,   25.00,    200_000, supports_vision=True, tier="flagship"),
    ModelInfo("azure/claude-haiku-4.5", "azure",    1.00,    5.00,    200_000, supports_vision=True, tier="budget"),

    # Azure-hosted xAI Grok
    ModelInfo("azure/grok-4.20",        "azure",    2.00,    6.00,  2_000_000, is_reasoning=True, tier="flagship"),
    ModelInfo("azure/grok-4",           "azure",    3.00,   15.00,    256_000, is_reasoning=True, tier="flagship"),
    ModelInfo("azure/grok-4.1-fast",    "azure",    0.20,    0.50,  2_000_000, tier="budget"),
    ModelInfo("azure/grok-3",           "azure",    3.00,   15.00,    131_000, tier="flagship"),
    ModelInfo("azure/grok-3-mini",      "azure",    0.30,    0.50,    131_000, tier="budget"),

    # Azure-hosted DeepSeek
    ModelInfo("azure/deepseek-v3.2",    "azure",    0.28,    0.42,    128_000, is_open_source=True, tier="budget"),
    ModelInfo("azure/deepseek-v3.1",    "azure",    0.27,    1.10,    128_000, is_open_source=True, tier="budget"),
    ModelInfo("azure/deepseek-r1",      "azure",    0.55,    2.19,    128_000, is_open_source=True, is_reasoning=True, tier="reasoning"),

    # Azure-hosted Meta Llama
    ModelInfo("azure/llama-4-maverick", "azure",    0.27,    0.85,  1_000_000, is_open_source=True, tier="budget"),
    ModelInfo("azure/llama-4-scout",    "azure",    0.08,    0.30, 10_000_000, is_open_source=True, tier="budget"),
    ModelInfo("azure/llama-3.3-70b",    "azure",    0.18,    0.18,    131_000, is_open_source=True, tier="budget"),

    # Azure-hosted Mistral
    ModelInfo("azure/mistral-large-3",  "azure",    2.00,    6.00,    128_000, is_open_source=True, tier="mid"),
    ModelInfo("azure/mistral-small",    "azure",    0.10,    0.30,    128_000, is_open_source=True, tier="budget"),
    ModelInfo("azure/codestral",        "azure",    0.30,    0.90,    262_000, is_open_source=True, tier="budget"),
    ModelInfo("azure/mistral-nemo",     "azure",    0.02,    0.04,    128_000, is_open_source=True, tier="budget"),

    # Azure-hosted Cohere
    ModelInfo("azure/command-r-plus",   "azure",    2.50,   10.00,    128_000, tier="mid"),
    ModelInfo("azure/command-r",        "azure",    0.50,    1.50,    128_000, tier="budget"),

    # Azure-hosted Microsoft Phi
    ModelInfo("azure/phi-4",            "azure",    0.07,    0.14,    128_000, is_open_source=True, tier="budget"),
    ModelInfo("azure/phi-4-mini",       "azure",    0.02,    0.04,    128_000, is_open_source=True, tier="budget"),
    ModelInfo("azure/phi-4-multimodal", "azure",    0.07,    0.14,    128_000, is_open_source=True, supports_vision=True, tier="budget"),
    ModelInfo("azure/mai-ds-r1",        "azure",    0.55,    2.19,    128_000, is_reasoning=True, tier="reasoning"),

    # Azure-hosted Moonshot Kimi
    ModelInfo("azure/kimi-k2.5",        "azure",    0.20,    0.80,  1_000_000, is_open_source=True, tier="budget"),
    ModelInfo("azure/kimi-k2-thinking", "azure",    0.20,    0.80,  1_000_000, is_open_source=True, is_reasoning=True, tier="reasoning"),

    # Azure-hosted GLM
    ModelInfo("azure/glm-5",            "azure",    0.50,    1.00,    128_000, is_open_source=True, tier="mid"),
)


# ══════════════════════════════════════════════════════
# AWS Bedrock models (accessed via "bedrock/" prefix)
# ══════════════════════════════════════════════════════

_register(
    ModelInfo("bedrock/claude-opus-4.6",    "bedrock", 5.00,  25.00, 1_000_000, supports_vision=True, tier="flagship"),
    ModelInfo("bedrock/claude-sonnet-4.6",  "bedrock", 3.00,  15.00, 1_000_000, supports_vision=True, tier="mid"),
    ModelInfo("bedrock/claude-haiku-4.5",   "bedrock", 1.00,   5.00,   200_000, supports_vision=True, tier="budget"),
    ModelInfo("bedrock/llama-4-maverick",   "bedrock", 0.27,   0.85, 1_000_000, is_open_source=True, tier="budget"),
    ModelInfo("bedrock/llama-4-scout",      "bedrock", 0.08,   0.30,10_000_000, is_open_source=True, tier="budget"),
    ModelInfo("bedrock/llama-3.3-70b",      "bedrock", 0.18,   0.18,   131_000, is_open_source=True, tier="budget"),
    ModelInfo("bedrock/mistral-large-3",    "bedrock", 2.00,   6.00,   128_000, is_open_source=True, tier="mid"),
    ModelInfo("bedrock/nova-pro",           "bedrock", 0.80,   3.20,   300_000, supports_vision=True, tier="mid"),
    ModelInfo("bedrock/nova-lite",          "bedrock", 0.06,   0.24,   300_000, supports_vision=True, tier="budget"),
    ModelInfo("bedrock/nova-micro",         "bedrock", 0.035,  0.14,   128_000, tier="budget"),
    ModelInfo("bedrock/command-r-plus",     "bedrock", 2.50,  10.00,   128_000, tier="mid"),
    ModelInfo("bedrock/command-r",          "bedrock", 0.50,   1.50,   128_000, tier="budget"),
    ModelInfo("bedrock/jamba-2-large",      "bedrock", 2.00,   8.00,   256_000, tier="mid"),
    ModelInfo("bedrock/jamba-2-mini",       "bedrock", 0.20,   0.40,   256_000, tier="budget"),
)


# ══════════════════════════════════════════════════════
# Query helpers
# ══════════════════════════════════════════════════════

def get_model(model_id: str) -> ModelInfo | None:
    """Look up a model by ID. Returns None if not found."""
    return MODELS.get(model_id)


def list_models(
    *,
    provider: str | None = None,
    tier: str | None = None,
    max_input_cost: float | None = None,
    supports_tools: bool | None = None,
    supports_vision: bool | None = None,
    is_reasoning: bool | None = None,
    is_open_source: bool | None = None,
    min_context: int | None = None,
) -> list[ModelInfo]:
    """Filter and list models from the registry."""
    results = list(MODELS.values())

    if provider:
        results = [m for m in results if m.provider == provider]
    if tier:
        results = [m for m in results if m.tier == tier]
    if max_input_cost is not None:
        results = [m for m in results if m.input_cost <= max_input_cost]
    if supports_tools is not None:
        results = [m for m in results if m.supports_tools == supports_tools]
    if supports_vision is not None:
        results = [m for m in results if m.supports_vision == supports_vision]
    if is_reasoning is not None:
        results = [m for m in results if m.is_reasoning == is_reasoning]
    if is_open_source is not None:
        results = [m for m in results if m.is_open_source == is_open_source]
    if min_context is not None:
        results = [m for m in results if m.context_window >= min_context]

    return sorted(results, key=lambda m: m.input_cost)


def list_providers() -> list[str]:
    """List all unique provider names."""
    return sorted(set(m.provider for m in MODELS.values()))


def cheapest_for_tier(tier: str) -> ModelInfo | None:
    """Find the cheapest model in a given tier."""
    candidates = [m for m in MODELS.values() if m.tier == tier]
    return min(candidates, key=lambda m: m.input_cost) if candidates else None


def estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost for a given model and token counts. Returns 0.0 if model unknown."""
    model = MODELS.get(model_id)
    if not model:
        return 0.0
    return (input_tokens * model.input_cost + output_tokens * model.output_cost) / 1_000_000
