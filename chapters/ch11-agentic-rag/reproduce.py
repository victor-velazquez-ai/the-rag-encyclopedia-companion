"""Chapter 11 reproduction — entity-resolution accuracy on alias pairs (offline, no API key).

The chapter's differentiator is the canonical-ID layer, not the agentic loop. This runnable check
builds a small crosswalk and scores the matcher's deterministic-then-fuzzy stack on labeled alias
pairs: each surface form should resolve to its true canonical id (or to None for an out-of-corpus
mention). It reports accuracy and, separately, the false-merge count - because over-merging is the
asymmetric error the chapter says to fear most.

The agentic loop (AgenticRAG) and the keyed multi-hop comparison need a generation key; this offline
slice exercises the precision substrate that runs on every query.

    python chapters/ch11-agentic-rag/reproduce.py
"""

from __future__ import annotations

from ragkit.architectures.agentic import EntityResolver


def _build_resolver() -> EntityResolver:
    r = EntityResolver()
    r.add_canonical("acme", "Acme Inc", "Acme, Inc.", "ACME Corporation")
    r.add_canonical("ibm", "IBM", "International Business Machines")
    r.add_canonical("globex", "Globex", "Globex Corp")
    return r


# (surface form, expected canonical id or None)
_CASES: list[tuple[str, str | None]] = [
    ("acme inc", "acme"),          # deterministic (normalized exact)
    ("ACME, Inc.", "acme"),        # deterministic
    ("I.B.M.", "ibm"),             # fuzzy (despaced acronym)
    ("International Business Machines", "ibm"),  # deterministic
    ("Globex Corporation", "globex"),  # deterministic (suffix-stripped)
    ("Initech", None),             # clean miss (out of corpus)
    ("Umbrella Corp", None),       # clean miss
]


def main() -> None:
    resolver = _build_resolver()
    correct = 0
    false_merges = 0  # resolved to an id when the truth is None, or to the WRONG id
    print("surface form                         expected   got        ok")
    print("-" * 64)
    for surface, expected in _CASES:
        got = resolver.resolve(surface)
        ok = got == expected
        correct += ok
        if not ok and got is not None:
            false_merges += 1
        print(f"{surface:<36} {str(expected):<10} {str(got):<10} {'OK' if ok else 'X'}")

    n = len(_CASES)
    print("-" * 64)
    print(f"accuracy:      {correct}/{n} = {correct / n:.0%}")
    print(f"false merges:  {false_merges} (resolved to a wrong/spurious id - the asymmetric error)")
    assert correct == n, "entity-resolution accuracy regressed"
    assert false_merges == 0, "a false merge slipped through - over-merging is the costly error"
    print("\nPASS - deterministic-then-fuzzy resolves every alias and merges nothing it should not.")


if __name__ == "__main__":
    main()
