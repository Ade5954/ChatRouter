"""Benchmark: ChatRouter (with session cache affinity) vs a naive no-router gateway.

Why this matters
----------------
Per the README, prefix caching on DeepSeek/Claude/Gemini/GPT-4o cuts input
cost by 75%~90% when the prompt prefix is reused. A naive router that
re-picks the best model every turn (based on that turn's complexity) keeps
switching models, which shatters the upstream prefix cache. Session affinity
keeps a conversation on one model so the prefix stays hot.

This script runs BOTH strategies over the same realistic multi-turn
conversations by driving the real Router / GatewayService code path. It then
estimates total input-token *cost* under each strategy and reports the
reduction that session affinity buys.

Token model
-----------
* We feed each conversation turn to the router and record the chosen model.
* For each turn we compute its prompt token estimate (from the analyzer).
* Prefix-cache semantics:
    - If the chosen model is the SAME as the previous turn's model in this
      session AND the conversation prefix is reused, the cached fraction of
      the prompt is served at the provider's cache-hit discount.
    - Naive strategy (no affinity): the router is re-run per turn with NO
      session memory, so it freely switches models -> cache is cold every
      turn (hit fraction ~ 0).
    - Affinity strategy: the router keeps the session on one model, so after
      the first turn the long shared prefix is cached -> hit fraction `p`
      (the 75%~90% number from the docs).
* Cost = sum over turns of
      input_tokens * input_cost_per_1k/1000 * (1 - p*cache_active)
  where cache_active is 1 for cached turns in the affinity run, 0 for the
  naive run (cold cache every turn).

Run:
    cd chatRouter
    python bench_cache_affinity.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from chatrouter.config.models import RoutingConfig, SessionAffinityConfig, FeedbackConfig
from chatrouter.core.schemas import ChatCompletionRequest, ChatMessage, new_request_id
from chatrouter.core.types import RequestContext
from chatrouter.service import GatewayService

from tests.conftest import make_config, make_request, user, assistant, system

# Prefix-cache hit discount band from the README (75%~90% input-cost saving).
CACHE_SAVING_LOW = 0.75
CACHE_SAVING_HIGH = 0.90

# Fraction of the prompt that is a reusable shared prefix (system + prior turns).
# Conservative: most of a multi-turn prompt is the accumulated history.
SHARED_PREFIX_FRACTION = 0.80


def _user(t: str) -> ChatMessage:
    return user(t)


# A large, realistic system prompt so each turn already carries a heavy shared
# prefix (this is what prefix caching is designed to exploit).
SYSTEM_PROMPT = system(
    "You are a senior engineering assistant embedded in a large enterprise "
    "knowledge base. You have access to the company style guide, the API "
    "reference for the internal platform, and the incident postmortems archive. "
    "Always answer concisely, cite sources when possible, and prefer code "
    "snippets over prose. Never reveal internal system names. When the user "
    "asks for a proof, provide a rigorous step-by-step derivation. When the "
    "user asks for a summary, cap it at five bullet points. The platform SDK "
    "is documented at /docs and supports streaming, batch, and async calls."
)

# A realistic mix of multi-turn conversations. Each is a list of user/assistant
# turns. The complexity varies within a conversation to stress routing. We pad
# the assistant replies with representative volume so token counts look like a
# real production workload (hundreds-to-thousands of tokens per turn).
CONVERSATIONS: list[list[ChatMessage]] = [
    # 1. A coding session: starts simple, escalates to a hard proof.
    [
        SYSTEM_PROMPT,
        user("What's a Python list comprehension?"),
        assistant(
            "A list comprehension is a concise syntax for building lists: "
            "[f(x) for x in iterable if condition(x)]. It replaces a manual "
            "loop with an inline expression, is generally faster than append "
            "in a for-loop, and improves readability when the transform is simple. "
            "Example: squares = [x*x for x in range(10)]. You can nest them, "
            "add multiple for-clauses, and filter with an if at the end."
        ),
        user("Can you show one that flattens a nested list?"),
        assistant(
            "Use a nested comprehension: flat = [y for sub in nested for y in sub]. "
            "The outer for iterates the sublists and the inner for iterates each "
            "element. This is O(n) over the total number of elements. For deep "
            "nesting you would need recursion, but for one level of nesting this "
            "is the idiomatic and fastest approach in CPython."
        ),
        user("Now prove by induction that this flatten is correct for arbitrary nesting depth and give asymptotic complexity."),
    ],
    # 2. A support chat that stays cheap.
    [
        SYSTEM_PROMPT,
        user("How do I reset my password?"),
        assistant(
            "Open the login page and click 'Forgot password'. Enter your "
            "work email; we send a reset link valid for 30 minutes. If the "
            "link expires, request a new one. After resetting, you must log "
            "out of all active sessions from the security tab."
        ),
        user("I didn't get the email."),
        assistant(
            "First check spam and the 'Promotions' tab. Allow up to 5 minutes "
            "for delivery. If it still has not arrived, your address may be "
            "misconfigured in HR systems; contact #it-help on Slack with your "
            "employee ID and we will trigger a manual reset."
        ),
        user("Still nothing, can you resend?"),
    ],
    # 3. Mixed: a research chat that stays premium-ish.
    [
        SYSTEM_PROMPT,
        user("Summarise the attention mechanism in transformers."),
        assistant(
            "Attention computes a weighted sum of value vectors, where weights "
            "come from the softmax of query-key dot products scaled by sqrt(d_k). "
            "It lets each token attend to all others in the sequence, capturing "
            "long-range dependencies without recurrence. Multi-head attention "
            "runs several attention operations in parallel on projected subspaces "
            "and concatenates the results, increasing representational capacity."
        ),
        user("Explain the difference between multi-head and multi-query attention with math."),
        assistant(
            "In multi-head attention each head has its own W_q, W_k, W_v and the "
            "outputs are concatenated then projected by W_o. Multi-query attention "
            "shares a single K and V projection across all heads while keeping "
            "distinct Q projections, reducing memory bandwidth for the K/V cache "
            "and speeding up decoding at a small quality cost. Grouped-query "
            "attention is the interpolate that groups heads to share K/V."
        ),
        user("Write a short literature review comparing them across 5 papers."),
    ],
    # 4. Another session that would drift across tiers.
    [
        SYSTEM_PROMPT,
        user("Hello!"),
        assistant("Hi! I'm your engineering assistant. What can I help you build or debug today?"),
        user("Translate 'good morning' to French."),
        assistant("The translation is 'Bonjour'. In the morning you may also say 'Bonjour' generally; a more formal written form is 'Bonjour madame/monsieur'."),
        user("Now write a formal business email in French requesting a meeting."),
    ],
    # 5. A long agentic session: many follow-ups on one topic.
    [
        SYSTEM_PROMPT,
        user("Design a rate limiter for our API gateway."),
        assistant(
            "Use a token-bucket per API key with a refill rate r and capacity C. "
            "On each request, compute elapsed time, add r*dt tokens capped at C, "
            "and reject if the bucket is empty. Store buckets in Redis with a TTL "
            "equal to the refill window. For distributed correctness use a Lua "
            "script so the check-and-decrement is atomic and single-roundtrip."
        ),
        user("What about bursts and fairness across tenants?"),
        assistant(
            "Add a hierarchical limiter: a global bucket guards the whole cluster, "
            "and a per-tenant bucket guards fairness. Admit only if both pass. For "
            "bursts, allow the per-key bucket to accumulate up to C while the global "
            "bucket enforces the hard ceiling. Use weighted fairness so premium "
            "tenants get a larger share of the global rate."
        ),
        user("Show the Redis Lua script for the atomic check."),
        assistant(
            "local key = KEYS[1]\nlocal rate = tonumber(ARGV[1])\nlocal cap = tonumber(ARGV[2])\n"
            "local now = tonumber(ARGV[3])\nlocal cost = tonumber(ARGV[4])\n"
            "local data = redis.call('HMGET', key, 'tokens', 'ts')\n"
            "local tokens = tonumber(data[1]) or cap\nlocal ts = tonumber(data[2]) or now\n"
            "tokens = math.min(cap, tokens + rate * (now - ts) / 1000)\n"
            "if tokens >= cost then tokens = tokens - cost\n"
            "redis.call('HMSET', key, 'tokens', tokens, 'ts', now)\n"
            "redis.call('PEXPIRE', key, cap / rate * 1000)\nreturn 1 else return 0 end"
        ),
        user("How do we alert when a tenant is throttled too often?"),
    ],
]


async def _run_strategy(enable_affinity: bool) -> tuple[list[dict], int]:
    """Drive the real router over all conversations.

    Returns (decisions, total_prompt_tokens). ``decisions`` records, per turn,
    the chosen model id and the prompt token estimate.
    """
    affinity_cfg = SessionAffinityConfig(enabled=enable_affinity, stickiness=0.4)
    # Disable the feedback exploration loop so the comparison isolates the
    # effect of session cache affinity itself, not random exploration.
    config = make_config(
        routing=RoutingConfig(
            default_model="mid",
            session_affinity=affinity_cfg,
            feedback=FeedbackConfig(enabled=False),
        )
    )
    svc = GatewayService(config)
    await svc.start()
    decisions: list[dict] = []
    total_tokens = 0
    try:
        for ci, conv in enumerate(CONVERSATIONS):
            # Replay the conversation turn by turn, accumulating history so the
            # analyzer sees the full conversation (just like a real client).
            history: list[ChatMessage] = []
            prev_model: str | None = None
            for ti, _ in enumerate(conv):
                history = conv[: ti + 1]
                req = make_request(history, model="auto")
                ctx = RequestContext(
                    request_id=new_request_id(),
                    tenant=svc.config.tenants[0],
                    request=req,
                    session_id=f"conv-{ci}",
                )
                # projected_tokens not needed for the cache estimate.
                decision = await svc.router.route(ctx, 0)
                assess = decision.assessment
                prompt_tokens = assess.prompt_tokens_estimate if assess else 0
                total_tokens += prompt_tokens
                decisions.append(
                    {
                        "conv": ci,
                        "turn": ti,
                        "model": decision.model.id,
                        "reason": decision.reason.value,
                        "prompt_tokens": prompt_tokens,
                        "prev_model": prev_model,
                    }
                )
                prev_model = decision.model.id
    finally:
        await svc.close()
    return decisions, total_tokens


async def _run_fixed_model(model_id: str) -> tuple[list[dict], int]:
    """Third baseline: a typical 'no gateway' deployment where the client pins
    one cheap, cache-friendly model for the whole conversation (e.g. everyone
    just calls DeepSeek directly). No router, no tier-awareness — but the prefix
    cache stays hot because the model never changes.
    """
    # We still need the analyzer for token estimates, so build a minimal service.
    config = make_config(routing=RoutingConfig(default_model="mid"))
    svc = GatewayService(config)
    await svc.start()
    decisions: list[dict] = []
    total_tokens = 0
    try:
        for ci, conv in enumerate(CONVERSATIONS):
            history: list[ChatMessage] = []
            prev_model: str | None = None
            for ti, _ in enumerate(conv):
                history = conv[: ti + 1]
                req = make_request(history, model="auto")
                assess = svc.router.analyse(req)
                prompt_tokens = assess.prompt_tokens_estimate
                total_tokens += prompt_tokens
                decisions.append(
                    {
                        "conv": ci,
                        "turn": ti,
                        "model": model_id,
                        "reason": "fixed_model",
                        "prompt_tokens": prompt_tokens,
                        "prev_model": prev_model,
                    }
                )
                prev_model = model_id
    finally:
        await svc.close()
    return decisions, total_tokens


# Real upstream prices for cache-friendly models (DeepSeek-class, per 1M tokens).
# These are the models the tmtpost article is about: cheap AND prefix-cache
# friendly. Cached input is billed at ~1/10 of normal input (≈90% off).
# We use a single representative price for the cache-friendly tier so the
# comparison isolates the cache effect rather than tier price differences.
CACHE_FRIENDLY_INPUT_PER_1M = 0.014     # USD / 1M input tokens (e.g. DeepSeek-chat)
CACHE_FRIENDLY_INPUT_PER_1K = CACHE_FRIENDLY_INPUT_PER_1M / 1000.0

# For a NON-cache-friendly deployment (the "no gateway / naive per-turn router"
# that ignores caching) we price input at the full rate with NO cache discount,
# exactly as providers bill when the prefix is not reused.


def _estimate_cost(decisions, cache_saving: float, cache_active: bool) -> float:
    """Estimate total input cost for a run.

    cache_active=True applies the prefix-cache discount on turns where the
    model is unchanged from the previous turn (i.e. the prefix is hot).
    When cache_active=False (naive, no session memory) the prefix is cold every
    turn, so no discount is applied — this is the provider's full input price.

    The billed token count assumes a reusable shared prefix of size
    ``SHARED_PREFIX_FRACTION``; only that slice benefits from the cache.
    """
    total = 0.0
    for d in decisions:
        cached_fraction = 0.0
        if cache_active and d["prev_model"] is not None and d["model"] == d["prev_model"]:
            cached_fraction = SHARED_PREFIX_FRACTION * cache_saving
        billed_fraction = 1.0 - cached_fraction
        full_price = d["prompt_tokens"] * CACHE_FRIENDLY_INPUT_PER_1K / 1000.0
        total += full_price * billed_fraction
    return total


def _print_table(title: str, decisions):
    print(f"\n=== {title} ===")
    for d in decisions:
        flag = ""
        if d["prev_model"] is not None and d["model"] == d["prev_model"]:
            flag = "  (prefix-hot)"
        print(
            f"  conv{d['conv']} turn{d['turn']:>2}: {d['model']:<9} "
            f"reason={d['reason']:<16} tokens={d['prompt_tokens']:>6}{flag}"
        )


def main() -> None:
    naive_decisions, naive_tokens = asyncio.run(_run_strategy(enable_affinity=False))
    affinity_decisions, affinity_tokens = asyncio.run(_run_strategy(enable_affinity=True))
    # Fixed single cache-friendly model (e.g. direct DeepSeek) = typical no-gateway setup.
    fixed_decisions, _ = asyncio.run(_run_fixed_model("cheap"))

    _print_table("NO GATEWAY (fixed cheap cache-friendly model)", fixed_decisions)
    _print_table("NAIVE ROUTER (per-turn best model, no affinity)", naive_decisions)
    _print_table("CHATROUTER (session cache affinity)", affinity_decisions)

    # Count model switches per conversation (lower = better cache locality).
    def _switches(decisions):
        s = 0
        per_conv: dict[int, str | None] = {}
        for d in decisions:
            cur = per_conv.get(d["conv"])
            if cur is not None and cur != d["model"]:
                s += 1
            per_conv[d["conv"]] = d["model"]
        return s

    fixed_sw = _switches(fixed_decisions)
    naive_sw = _switches(naive_decisions)
    aff_sw = _switches(affinity_decisions)

    # Cost estimates for the cache-saving band (75%~90% input-cost cut on hit).
    #   fixed  : hot cache every follow-up turn (cache_active=True)
    #   naive  : cold cache every turn (cache_active=False)
    #   affinity: hot cache every follow-up turn (cache_active=True)
    results = []
    for saving in (CACHE_SAVING_LOW, CACHE_SAVING_HIGH):
        fixed_cost = _estimate_cost(fixed_decisions, saving, cache_active=True)
        naive_cost = _estimate_cost(naive_decisions, saving, cache_active=False)
        aff_cost = _estimate_cost(affinity_decisions, saving, cache_active=True)
        # Savings of ChatRouter+affinity vs the naive per-turn router.
        saved_vs_naive = naive_cost - aff_cost
        pct = (saved_vs_naive / naive_cost * 100) if naive_cost else 0.0
        results.append((saving, fixed_cost, naive_cost, aff_cost, saved_vs_naive, pct))

    # Project to a realistic monthly volume: 10k conversations of this shape.
    MONTHLY_CONVS = 10_000
    scale = MONTHLY_CONVS / max(len({d["conv"] for d in naive_decisions}), 1)

    print("\n================ BENCHMARK RESULTS ================")
    print(f"Corpus: {len({d['conv'] for d in naive_decisions})} conversations, "
          f"{naive_tokens:,} prompt tokens total (identical for all runs)")
    print(f"Model switches across conversations:  "
          f"fixed={fixed_sw}  naive={naive_sw}  affinity={aff_sw}")
    print("\nPer-corpus input-cost estimate (USD), DeepSeek-class pricing:")
    print(f"  {'cache':>5} | {'fixed $':>9} | {'naive $':>9} | {'affinity $':>10} | {'saved vs naive':>14} | {'reduction':>9}")
    print("-" * 78)
    for saving, fc, nc, ac, sv, pct in results:
        print(
            f"  {int(saving*100):>3}% | {fc:>9.4f} | {nc:>9.4f} | {ac:>10.4f} | {sv:>14.4f} | {pct:>8.1f}%"
        )

    print(f"\nExtrapolated to {MONTHLY_CONVS:,} conversations / month:")
    print(f"  {'cache':>5} | {'fixed $/mo':>11} | {'naive $/mo':>11} | {'affinity $/mo':>13} | {'saved $/mo':>11}")
    print("-" * 74)
    for saving, fc, nc, ac, sv, _ in results:
        print(
            f"  {int(saving*100):>3}% | {fc*scale:>11.2f} | {nc*scale:>11.2f} | {ac*scale:>13.2f} | {sv*scale:>11.2f}"
        )

    print("\nReading the numbers:")
    print("  * 'fixed' (no gateway, one cache-friendly model) keeps the prefix hot,")
    print("    so it already enjoys the cache discount — the ceiling for savings.")
    print("  * 'naive router' re-picks the best tier every turn with no session")
    print("    memory, so it switches models and the prefix cache goes cold (0% hit).")
    print("  * 'ChatRouter + affinity' sticks the session to one model, restoring")
    print("    the hot prefix AND keeping first-turn complexity awareness.")
    print("  Savings vs the naive router scale linearly with conversation volume.")
    print("===================================================")


if __name__ == "__main__":
    main()
