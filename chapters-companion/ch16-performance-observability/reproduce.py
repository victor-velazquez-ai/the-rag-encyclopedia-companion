"""Chapter 16 reproduction — semantic-cache hit rate + PSI drift (offline, no API key).

The cost/observability levers are pure, so they reproduce with synthetic vectors and no embedding
key. This check measures a semantic cache's hit rate over a small query stream (a near-duplicate
hits, an unrelated query misses) and confirms PSI separates a no-shift sample (~0, band "none") from
a shifted one (> 0.2, band "significant"). Model routing is exercised too: simple to cheap, complex
to strong.

The keyed companion sweeps the static threshold against a vCache-style learned one and prices a
cheap/strong routing cascade; this offline slice verifies the levers behave.

    python chapters-companion/ch16-performance-observability/reproduce.py
"""

from __future__ import annotations

import random

from professional_rag_kit.production.observability import psi, psi_alert
from professional_rag_kit.production.serving import ModelRouter, lookup

# Cached (vector, response) entries - synthetic, so no embedder/key is needed.
_CACHE = [
    ([1.0, 0.0, 0.0], "Paris is the capital of France."),
    ([0.0, 1.0, 0.0], "Water boils at 100 C at sea level."),
]
# A query stream as (vector, should_hit) - near-duplicates hit, unrelated misses.
_STREAM = [
    ([0.97, 0.08, 0.0], True),
    ([0.05, 0.99, 0.02], True),
    ([0.10, 0.10, 0.99], False),
    ([0.20, 0.18, 0.96], False),
]


def main() -> None:
    print("Chapter 16 - semantic-cache hit rate + PSI drift (offline)")
    print("-" * 58)

    # --- Semantic cache: measure hit rate at a static threshold ---
    threshold = 0.85
    hits = 0
    for vec, should_hit in _STREAM:
        got = lookup(vec, _CACHE, threshold=threshold)
        is_hit = got is not None
        hits += is_hit
        assert is_hit == should_hit, "cache lookup disagreed with the labeled stream"
    rate = hits / len(_STREAM)
    print(f"cache: {hits}/{len(_STREAM)} hits = {rate:.0%} at threshold {threshold}")
    print("       (static threshold is a CORRECTNESS control, not a knob - see vCache)")

    # --- Model routing: simple vs complex ---
    router = ModelRouter()
    assert router.route("What time is it in Tokyo?") == router.cheap_model
    assert router.route(
        "Compare dense and sparse retrieval and explain when each wins."
    ) == router.strong_model
    print("routing: simple -> cheap, complex -> strong (RouteLLM/FrugalGPT shape)")

    # --- PSI drift: no-shift ~0 vs a large shift > 0.2 (seeded Gaussian samples) ---
    rng = random.Random(0)
    ref = [rng.gauss(0.0, 1.0) for _ in range(2000)]
    no_shift = [rng.gauss(0.0, 1.0) for _ in range(2000)]
    big_shift = [rng.gauss(1.0, 1.0) for _ in range(2000)]
    p_none, p_big = psi(ref, no_shift), psi(ref, big_shift)
    print(f"PSI: no-shift={p_none:.4f} ({psi_alert(p_none)}), "
          f"big-shift={p_big:.4f} ({psi_alert(p_big)})")
    assert p_none < 0.1 and psi_alert(p_none) == "none"
    assert p_big > 0.2 and psi_alert(p_big) == "significant"

    print("-" * 58)
    print("PASS - cache hit rate measured, routing routes, PSI flags the shift.")


if __name__ == "__main__":
    main()
