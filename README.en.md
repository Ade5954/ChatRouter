English · [中文](README.md)

# ChatRouter

A production-oriented, OpenAI-compatible LLM traffic gateway focused on **intelligent request routing** and **fine-grained traffic governance**.

As a drop-in replacement for the OpenAI SDK, ChatRouter manages a multi-vendor model pool behind a single entry point, continuously driving down inference cost while preserving service quality.

---

## Two Core Capabilities

### 1. Full-Conversation Context-Aware Routing

Naive **single-turn routers** score only the **latest user turn**, which systematically under-estimates difficulty in real multi-turn conversations:

| Scenario | Single-turn router's guess | Actual need |
|----------|---------------------------|-------------|
| `"And what about the other case?"` | Very short → cheap model | Inherits the complex derivation from earlier context |
| system prompt: "all answers must be rigorous proofs" | Later questions look simple | Strong reasoning required throughout |
| User replies "still wrong" three times in a row | Each turn looks short in isolation | Current tier is disproven, must upgrade |
| A 40-turn architecture discussion | Final turn is just a follow-up | Must satisfy all accumulated constraints |

> The mis-estimation above is a limitation of **single-turn routing**. ChatRouter reaches conversation-awareness with **explainable rule signals + recency weighting**, requiring no training data and yielding auditable, per-request decisions.

ChatRouter scores the **full conversation history** and avoids context-task mismatch through three mechanisms:

- **Recency weighting**: newer turns weigh more (`recency_decay`), but historical turns always retain influence, so an upfront hard requirement is never forgotten.
- **Escalation memory** (`escalation_memory`): once a thread shows high-difficulty traits, the weighted mean is pulled back toward the historical peak — a hard task won't be quietly downgraded just because the user said "ok".
- **Anaphoric inheritance**: follow-ups with no standalone meaning, such as `"continue"` or `"what about the other one"`, directly inherit the complexity of the preceding context.

Ten signal dimensions participate in scoring, with Chinese/English recognition: reasoning keywords, code content, conversation depth, context pressure, unresolved thread (user-dissatisfaction signal), tool calls, structured output, instruction density, expected output length, and multilingual mixing.

> **Aggregation**: the score is not a simple weighted average. Most signals are zero for any single request, so a pure mean would dilute a decisive signal like "prove this theorem" down into the cheap tier. The actual formula is `dominant signal × 0.6 + weighted mean × 0.4 + interaction term`, ensuring a single decisive signal can dominate while "several moderate signals stacking up" is also recognized as high difficulty.

`/v1/routing/explain` surfaces the complete decision rationale without calling any model.

### 1b. Session-Level Cache Affinity (the routing × caching trade-off)

Routing by complexity saves tokens, but **switching models too often shatters the upstream prefix cache** — and for cache-friendly models (DeepSeek, Claude, Gemini, GPT-4o-class), a prefix hit cuts input cost by **75–90%**, often more than routing saves. If every turn hops to a differently-sized model, that large caching benefit is wasted.

ChatRouter therefore ships **session-level cache affinity**: follow-up requests of the same `session_id` stay on the model already in use, unless the task's complexity drifts across more than `max_drift_tiers` (default 1 tier), in which case it upgrades/downgrades. Two mechanisms enforce it:

- **Utility penalty (dynamic, real cache loss)**: the switching penalty is no longer a hard-coded constant; it is computed in real time from the cost formula in [SeqRoute (arXiv 2026)](https://arxiv.org/abs/2602.11688) — leaving the session's current model forfeits the upstream prefix cache, and the one-time switch loss equals `historical prefix tokens × (c_in − c_cache)` USD (where `c_cache` comes from the model's `cached_input_cost_per_1k`, falling back to `c_in` when unset, i.e. no extra penalty). That loss is converted into the router's utility scale by a single constant and scaled by `stickiness` (default 0.4); setting it to 0 disables affinity, so models with a larger cache price gap (e.g. GPT-4o, DeepSeek) are naturally stickier while cheap models stay freely routable.
- **Hard-preference override**: if the session is already bound to a model whose tier is within the allowed drift of the target tier, that model is taken as the winner outright (`reason=session_affinity`), exploration is skipped, and the choice is written back for the next turn.

Boundaries: for **stable multi-turn sessions / high-cache-hit models**, stickiness is almost always better; for **one-shot requests** or **conversations with violent complexity jumps**, affinity automatically yields to the true complexity. Toggle or tune it via `routing.session_affinity`; each model's `cached_input_cost_per_1k` (cached-input price, e.g. ~0.5–0.1× the normal input price for OpenAI/DeepSeek) controls the stickiness strength.

### 2. Online Feedback-Loop Adaptive Routing

The routing policy does not stay frozen at the prior values written in config; it continuously self-iterates from **real production data**:

- **Explicit feedback**: `/v1/feedback` accepts thumbs up/down, 1–5 star ratings, and accept/reject flags.
- **Implicit signals**: retry counts, output truncation (`finish_reason=length`), latency regressions vs. baseline, and upstream failures are all automatically folded into quality evidence.
- **Tier-stratified statistics**: stats accumulate per `(model, complexity tier)`. A model may excel at simple tasks but clearly weaken on strong-reasoning tasks — a global mean would hide this; tier-stratified stats do not.
- **Confidence weighting**: the more samples, the stronger measured quality overrides the prior. A model with only two ratings barely moves decisions; after hundreds of ratings it dominates.
- **Single-use**: each `request_id` can be scored only once; duplicate submissions are discarded (`accepted: false`), preventing feedback poisoning.
- **Evidence half-life**: evidence older than the statistics window decays exponentially, so last week's one-off outage doesn't permanently drag the model down.
- **Exploration**: with probability `exploration_ratio`, sample the sub-optimal candidate to avoid locking into a local optimum and keep generating usable evidence for long-tail models.

Effect: when a model's quality drops on a specific task type, the gateway automatically reduces its scheduling within dozens of requests — no manual config change required.

#### Feedback Normalization

Clients express satisfaction in many idioms: a direct `score`, a 1–5 star `rating`, a 👍/👎, or behavioural signals like `accepted` / `regenerated` / `edited`. `/v1/feedback` collapses these **heterogeneous shapes into a single `[0,1]` quality score** before it enters the statistics loop. The mapping is centralized in the configurable `routing.feedback.normalization`:

| Shape | Default mapping |
|-------|----------------|
| `score` | verbatim (already in `[0,1]`) |
| `rating` | `(rating-1)/4`, 1→0.0, 5→1.0 |
| `thumb: up/down` | `1.0` / `0.0` |
| `accepted: true/false` | `1.0` / `0.0` |
| `regenerated` | `0.2` (weak negative — regenerating means the first answer missed) |
| `edited` | `0.5` (moderate negative — the answer helped but needed work) |

Priority is `score > rating > thumb > accepted > regenerated > edited`: the more deliberate the signal, the higher its precedence. The normalization result is echoed back via the `source` field in the feedback response, so you can audit "where did this score come from". Centralizing the mapping in config (rather than hard-coding it in the request schema) means operators can tune it to the business without code changes, and every piece of feedback stays traceable.

---

## Traffic Governance

| Capability | Description |
|------------|-------------|
| **RPM / TPM rate limiting** | Tenant-level request-count and token dual-dimension limiting. Tokens are pre-deducted on estimate, then reconciled against actual usage after the response |
| **Concurrency control** | Tenant-level cap on in-flight requests |
| **Tenant quotas** | Hour/day/month windows covering request count, tokens, and dollar spend; over-quota supports `reject` or `downgrade` (continue on the cheapest tier) |
| **Load overflow scheduling** | When a model saturates, automatically overflow to a model with spare capacity; when all saturate, briefly queue instead of failing outright |
| **Failure degradation** | Circuit breaker isolates faulty upstreams (closed → open → half-open auto-probing); failed requests retry down the fallback chain |
| **Retry strategy** | Exponential backoff + jitter; 4xx client errors are not retried (a model swap would also fail); streaming responses are only retryable before the first byte is delivered |
| **Context-overflow fallback** | When a conversation exceeds every candidate model's window, degrade per `routing.context_overflow.strategy` instead of failing directly |

### Context-Overflow Strategies

A conversation outgrowing the window is normal in production, not an exception, so a clear degradation path is needed:

| Strategy | Behavior |
|----------|----------|
| `reject` | Returns HTTP 400 (`context_length_exceeded`). Retries and scaling won't help; let the client know the truth early |
| `largest_window` (default) | Route to the widest-window model, even if its tier isn't the best fit for the task complexity |
| `trim_history` | Trim the middle of the conversation to make it fit. **Lossy**, so it must be explicitly enabled |

Trimming observes two invariants: the **system prompt** and the **most recent turns** are never dropped — the former defines the model's behavioral constraints, the latter is what is actually being asked; dropping either makes the model silently answer a different question. The middle is removed oldest-to-newest, and an elision notice is inserted by default so the model doesn't misread the remaining turns as a continuous dialogue.

### Response Caching

For **non-streaming** requests whose every field matches exactly, a cache hit returns the previous result directly **without calling upstream**, saving one generation. Config key `routing.response_cache`:

| Field | Description |
|-------|-------------|
| `enabled` | Off by default; only participates when enabled |
| `ttl_seconds` | Cache entry lifetime (default 300s) |
| `bypass_hints` | Requests carrying these `chatrouter` hints skip the cache, default `["session_id"]` — multi-turn conversations depend on context state the cache cannot see |
| `excluded_tenants` | Tenants that never read/write the cache (e.g. audit/eval tenants that must always reach the model) |

The cache key includes **every field that affects generated text**: the resolved target model, the full `messages`, plus sampling params `temperature` / `top_p` / `max_tokens` / `stop` / `tools` / `tool_choice` / `response_format` / `seed` / `user` / `n`. Any difference counts as a distinct request, so A's answer is never mis-sent to B. Streaming responses are never cached because SSE delivers in real time.

A cache hit still runs the full accounting flow (quota, billing, feedback learning), so it is **indistinguishable** from a real generation for the rest of the gateway — it just incurs no upstream cost. The response header carries `x-chatrouter-cache: HIT` on a hit.

If the protected head/tail alone exceeds the budget, trimming does not pretend to succeed; it records the fact honestly in the decision `notes`.

---

## Quick Start

```bash
pip install -r requirements.txt
pip install -e .

cp config/config.example.yaml config/config.yaml
# edit config.yaml, configure your providers and model pool

export OPENAI_API_KEY=sk-...
export CHATROUTER_ADMIN_KEY=your-admin-key

python -m chatrouter --check    # validate config
python -m chatrouter            # start service
```

Docker:

```bash
docker compose up -d
```

### Calling

Point `base_url` at the gateway; everything else is identical to the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-chatrouter-dev")

# model="auto" triggers intelligent routing
response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Derive the time complexity of this recursion and prove its optimality"}],
)

print(response.model)  # the model that actually served the request
```

Submit feedback to drive policy iteration (`request_id` comes from the `x-chatrouter-request-id` response header):

```bash
curl -X POST http://localhost:8000/v1/feedback \
  -H "Authorization: Bearer sk-chatrouter-dev" \
  -H "Content-Type: application/json" \
  -d '{"request_id": "chatcmpl-rt-...", "thumb": "down"}'
```

Inspect the full routing decision without calling a model:

```bash
curl -X POST http://localhost:8000/v1/routing/explain \
  -H "Authorization: Bearer sk-chatrouter-dev" \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Prove this theorem"}]}'
```

---

## API Overview

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat completion (streaming / non-streaming), OpenAI-compatible |
| GET | `/v1/models` | Model list (filtered by tenant permissions) |
| GET | `/v1/models/{id}` | Single model detail |
| POST | `/v1/feedback` | Submit quality feedback |
| POST | `/v1/routing/explain` | Routing decision dry-run |
| GET | `/healthz` · `/readyz` | Liveness / readiness probes |
| GET | `/metrics` | Prometheus metrics |
| GET | `/admin/status` | Live load, circuit breakers, quotas, learned quality |
| GET | `/admin/config` | Effective config (credentials redacted) |

### Response Headers

| Header | Meaning |
|--------|---------|
| `x-chatrouter-request-id` | Request ID, used when submitting feedback |
| `x-chatrouter-model` | Model that actually served the request |
| `x-chatrouter-routing-reason` | Routing reason (`context_aware` / `feedback_adaptive` / `overflow` / `exploration` …) |
| `x-chatrouter-complexity` | Conversation complexity score `[0,1]` |
| `x-chatrouter-tier` | Assigned capability tier |
| `x-chatrouter-context-trimmed` | Number of messages trimmed to fit the window (only present when trimming happened) |
| `x-chatrouter-cache` | `HIT` on a cache hit; absent otherwise |
| `x-ratelimit-*` | Rate-limit remaining |

### Per-Request Routing Hints

Attach a `chatrouter` field to the request body to fine-tune policy for a single request:

```json
{
  "model": "auto",
  "messages": [...],
  "chatrouter": {
    "session_id": "conv-123",
    "min_tier": "premium",
    "prefer_models": ["gpt-4o"],
    "quality_bias": 0.9
  }
}
```

---

## Configuration Essentials

The model pool is split into four tiers by capability: `economy` → `standard` → `premium` → `reasoning`. The complexity score maps to a target tier via thresholds, then a utility function picks the best model within the tier:

```
utility = quality_preference × feedback-adjusted quality
        + cost_preference × cost_score
        + latency_preference × latency_score
        + load_score
        - tier_offset_penalty
        - health_penalty
        - session_affinity_cache_loss_penalty (historical prefix tokens × (c_in − c_cache), see above)
```

`routing.quality_bias` is the core knob: `0` is maximally cost-saving, `1` is maximally quality-focused, default `0.6`. Overridable per tenant.

Full configuration reference is in `config/config.example.yaml`, every field commented. Supports `${VAR}` and `${VAR:-default}` environment-variable expansion.

---

## Deployment Shapes

- **Single replica**: `storage.backend: memory`, zero external dependencies.
- **Multi replica**: `storage.backend: redis`. Rate-limit counters, quotas, and feedback stats are shared via Redis (counters use Lua scripts for atomicity); the gateway itself is stateless and horizontally scalable. Circuit-breaker state is deliberately kept in-process — each replica observes failures consistently, and local decisions are faster.

---

## Development

```bash
pip install -e ".[dev]"
pytest              # 167 tests
ruff check src tests
```

Test coverage: complexity analysis (including the key context-awareness assertions), routing decisions, feedback adaptation, rate limiting, quotas, circuit breaking, load overflow, config validation, and mock-upstream end-to-end HTTP flows (including streaming and failure degradation).

---

## Scope

This project **focuses solely on backend LLM traffic governance** and does not include: agent workflow orchestration, RAG retrieval orchestration, model training/fine-tuning, or front-end UI.

## References

- **SeqRoute: Global Budget-Aware Sequential LLM Routing via Offline Reinforcement Learning** (arXiv 2026) — cost modeling of the prefix-cache loss from session switching. This project upgrades the fixed stickiness switching penalty into a dynamic, real cache-loss computation based on it. <https://arxiv.org/abs/2602.11688>

## License

MIT
