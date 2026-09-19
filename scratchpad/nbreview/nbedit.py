#!/usr/bin/env python3
"""Helpers for editing notebooks by cell index, preserving everything else.

Every mutator asserts the anchor it expects, so an edit against a notebook whose
shape has changed fails loudly instead of silently landing in the wrong cell.
"""
import json


class NB:
    def __init__(self, name):
        self.path = name if name.endswith(".ipynb") else f"docs/notebooks/{name}.ipynb"
        with open(self.path) as fh:
            self.nb = json.load(fh)
        self.edits = 0

    def _lines(self, text):
        lines = text.split("\n")
        return [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

    def cell(self, i):
        return self.nb["cells"][i]

    def set_source(self, i, text, *, expect=None, kind=None):
        c = self.cell(i)
        cur = "".join(c["source"])
        if kind:
            assert c["cell_type"] == kind, f"cell {i} is {c['cell_type']}, expected {kind}"
        if expect is not None:
            assert expect in cur, f"cell {i} does not contain {expect!r}\n--- got ---\n{cur[:400]}"
        c["source"] = self._lines(text)
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None
        self.edits += 1

    def sub(self, i, old, new, *, count=1):
        """Substring replacement inside one cell."""
        c = self.cell(i)
        cur = "".join(c["source"])
        assert cur.count(old) >= count, f"cell {i}: {old!r} not found\n--- got ---\n{cur[:600]}"
        c["source"] = self._lines(cur.replace(old, new, count))
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None
        self.edits += 1

    def insert(self, i, kind, text):
        cell = {"cell_type": kind, "metadata": {}, "source": self._lines(text)}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        self.nb["cells"].insert(i, cell)
        self.edits += 1

    def delete(self, i, *, expect=None):
        cur = "".join(self.cell(i)["source"])
        if expect is not None:
            assert expect in cur, f"cell {i} does not contain {expect!r}"
        del self.nb["cells"][i]
        self.edits += 1

    def save(self):
        assert self.edits, "PROBE DID NOT FIRE: no edits applied"
        with open(self.path, "w") as fh:
            json.dump(self.nb, fh, indent=1, ensure_ascii=False)
            fh.write("\n")
        print(f"[nbedit] {self.path}: {self.edits} edit(s) applied")
