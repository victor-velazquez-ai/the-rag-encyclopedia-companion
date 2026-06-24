"""ragkit.eval.suite — the reproduction runner behind `make reproduce` (Book Ch 14).

`python -m ragkit.eval.suite --all` re-runs the book's head-to-head comparisons — every chapter's
`reproduce.py` experiment — on the version-controlled golden set, printing a *quality* number and a
*cost* number for each, because the book's argument is never quality alone. This is what makes every
verdict in the book falsifiable on your own corpus: the expensive path has to earn its place, here,
in a table you can read.

Phase-1 scaffold: prints a Phase-2 placeholder; the runner lands in Phase 2.
"""


def main() -> None:
    print("Phase 2 — reproduction suite")


if __name__ == "__main__":
    main()
