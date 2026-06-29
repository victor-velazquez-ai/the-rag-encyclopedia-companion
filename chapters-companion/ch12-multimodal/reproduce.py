"""Chapter 12 reproduction — visual document retrieval (extra-gated, runnable note).

The headline comparison (ColPali/ColQwen visual retrieval vs. an OCR-then-embed text pipeline on
ViDoRe V2) needs the optional `[multimodal]` extra - colpali-engine plus a GPU-backed host, because
page encoding is a VLM forward pass. There is no faithful laptop-only stand-in: a fake embedder would
demonstrate nothing about the 81.3 vs. 67.0 nDCG@5 delta that is the entire business case.

So this reproduction is extra-gated. It detects whether the extra is installed and prints either the
run plan (extra present) or the one-line install command (extra absent). It always exits cleanly.

    python chapters-companion/ch12-multimodal/reproduce.py
"""

from __future__ import annotations

import importlib.util


def main() -> None:
    have_extra = importlib.util.find_spec("colpali_engine") is not None
    print("Chapter 12 - visual document retrieval (ColPali/ColQwen)")
    print("-" * 60)
    if not have_extra:
        print("The [multimodal] extra is NOT installed, so the visual-retrieval")
        print("comparison cannot run. Install it on a GPU-backed host:")
        print()
        print("    pip install -e '.[multimodal]'")
        print()
        print("Then re-run. The comparison indexes ViDoRe V2 pages with")
        print("VisualRetriever vs. an OCR-then-embed text pipeline and reports")
        print("page-level nDCG@5 (target ~81.3 vs. ~67.0) plus storage per page.")
        return

    print("The [multimodal] extra IS installed. Run plan:")
    print("  1. Index ViDoRe V2 pages with VisualRetriever.default() (ColQwen2.5).")
    print("  2. Index the same pages via OCR-then-embed as the text baseline.")
    print("  3. Score both with nDCG@5 on the shared query set; report the delta.")
    print("  4. Pair each quality number with storage/page (visual ~30x text).")
    print()
    print("(Requires a GPU and the ViDoRe V2 page set; this is the wiring note.)")


if __name__ == "__main__":
    main()
