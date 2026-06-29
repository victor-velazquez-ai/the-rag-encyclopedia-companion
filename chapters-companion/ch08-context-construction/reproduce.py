"""Chapter 8 reproduction — the lost-in-the-middle fold, demonstrated (offline, no API key)

Holds the retrieved passage *set* fixed and varies only the assembly order: rank-order vs. the zipper
fold (rank 1 first, rank 2 last, zippering inward). Shows where each passage lands so you can see the
strongest evidence move onto the two attention peaks and the weakest into the dead middle. Measuring
the *quality* delta needs an LLM judge (a key); the reordering itself — the part that costs nothing —
is what you reclaim for free, shown here.

    python chapters-companion/ch08-context-construction/reproduce.py
"""

from professional_rag_kit.retrieval.context import reorder_lost_in_middle

RANKED = [f"R{i} (rank {i})" for i in range(1, 8)]  # a reranked shortlist, strongest first


def main() -> None:
    folded = reorder_lost_in_middle(RANKED)
    n = len(folded)
    print("\nAssembly order (set fixed, only the ORDER changes):\n")
    print("  position | rank-order      | zipper fold")
    print("  ---------|-----------------|-----------------")
    for i in range(n):
        peak = " <- attention peak" if i == 0 or i == n - 1 else ""
        print(f"  {i:>8} | {RANKED[i]:<15} | {folded[i]:<15}{peak}")
    print("\n  The fold puts rank 1 and rank 2 on the start/end peaks and buries the weakest passage")
    print("  in the low-attention middle. Zero added latency. Measuring the answer-quality lift")
    print("  needs an LLM judge (a key); the reordering is the free win.\n")


if __name__ == "__main__":
    main()
