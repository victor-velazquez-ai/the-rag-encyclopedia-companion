"""Chapter 2 experiment - offline parse round-trip sanity (the runnable core of the chapter).

The book's headline is a three-way parser comparison (detector-first hybrid vs. pure VLM vs.
classical OCR) whose quality/cost deltas require GPU services and a sample corpus. That full bench
is the chapter's frontier and runs against the external hybrid parser. What is *fully runnable with
no GPU, no key, no network* - and what this script reproduces - is the chapter's structural
guarantee: parsing must be FAITHFUL. Plain text and Markdown survive unmangled, and an HTML table
serializes to a Markdown grid (structure preserved), losslessly for rectangular tables and lossily
(by flattening spans) for merged cells - exactly as Ch 2 documents.

Run:  python chapters-companion/ch02-document-processing/reproduce.py
"""

from __future__ import annotations

from professional_rag_kit.ingestion.parsing import html_table_to_markdown, parse


def _check(name: str, ok: bool) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return ok


def main() -> int:
    print("Ch 2 - offline parse round-trip sanity\n")
    results: list[bool] = []

    # 1. Plain text passes through (only outer whitespace stripped).
    txt = "  Parsing is the first stage of RAG.  "
    results.append(_check("plain text preserved", parse(txt) == "Parsing is the first stage of RAG."))

    # 2. Markdown is the target format and must NOT be mangled.
    md = "# Methods\n\nWe report **recall@k** and `nDCG@10`."
    results.append(_check("markdown structure intact", parse(md, content_type="text/markdown") == md))

    # 3. A rectangular HTML table -> a Markdown pipe table (grid survives, values present).
    table = ("<table><tr><th>Model</th><th>License</th></tr>"
             "<tr><td>Qwen3</td><td>Apache-2.0</td></tr></table>")
    out = html_table_to_markdown(table)
    rectangular = out.count("|") >= 6 and "Qwen3" in out and "Apache-2.0" in out and "---" in out
    results.append(_check("HTML table -> rectangular Markdown grid", rectangular))

    # 4. A table inside an HTML page is inlined by parse() (sniffed as HTML, no content_type).
    page = "<html><body><p>Scores:</p><table><tr><th>A</th></tr><tr><td>1</td></tr></table></body></html>"
    inlined = "| A |" in parse(page) and "Scores:" in parse(page)
    results.append(_check("page-level table inlined to Markdown", inlined))

    # 5. The documented LOSSY edge: a colspan is flattened by repetition (keep source HTML for these).
    merged = "<table><tr><th colspan='2'>Q4</th></tr><tr><td>NA</td><td>412</td></tr></table>"
    flattened = html_table_to_markdown(merged).splitlines()[0].count("Q4") == 2
    results.append(_check("colspan flattened (lossy, as documented)", flattened))

    print()
    passed = sum(results)
    print(f"{passed}/{len(results)} checks passed.")
    print("Frontier (not run here): scanned PDFs / dense layouts need the external detector-first")
    print("hybrid VLM/OCR parser, and a VLM chart number must be VERIFIED before it becomes a fact.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
