"""py2nb — convert a percent-format script into a Jupyter notebook (pure Python, no deps).

Each walkthrough is authored as a runnable ``.py`` in *percent* format so the code is verified by
running the script; the matching ``.ipynb`` is generated from it, so the two never drift.

Cell markers:
    # %% [markdown]     → a markdown cell; following `# ...` comment lines are the markdown body
    # %%                → a code cell; following lines are code

Usage:  python tools/py2nb.py path/to/walkthrough.py   # writes walkthrough.ipynb next to it
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _split_cells(lines: list[str]):
    cells, kind, buf = [], "code", []

    def flush():
        if buf or cells:  # don't emit a leading empty cell
            cells.append((kind, buf.copy()))
        buf.clear()

    started = False
    for line in lines:
        stripped = line.rstrip("\n")
        if stripped.startswith("# %%"):
            if started:
                flush()
            started = True
            kind = "markdown" if "[markdown]" in stripped else "code"
            continue
        buf.append(line)
    if started:
        flush()
    return cells


def _to_source(lines: list[str], markdown: bool) -> list[str]:
    # drop leading/trailing blank lines
    while lines and lines[0].strip() == "":
        lines = lines[1:]
    while lines and lines[-1].strip() == "":
        lines = lines[:-1]
    if markdown:
        lines = [(ln[2:] if ln.startswith("# ") else ln.lstrip("#")) for ln in lines]
    # nbformat wants each element to retain its trailing newline except the last
    src = []
    for i, ln in enumerate(lines):
        text = ln.rstrip("\n")
        src.append(text + ("\n" if i < len(lines) - 1 else ""))
    return src


def convert(py_path: str) -> str:
    p = Path(py_path)
    raw = p.read_text(encoding="utf-8").splitlines(keepends=True)
    # strip a module docstring if the file starts with one (keep notebooks clean)
    cells = _split_cells(raw)
    nb_cells = []
    for kind, body in cells:
        source = _to_source(body, markdown=(kind == "markdown"))
        if not source:
            continue
        if kind == "markdown":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": source})
        else:
            nb_cells.append(
                {"cell_type": "code", "metadata": {}, "execution_count": None,
                 "outputs": [], "source": source}
            )
    nb = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = p.with_suffix(".ipynb")
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(out)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print("wrote", convert(arg))
