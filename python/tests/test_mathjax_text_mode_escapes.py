"""No authored page may escape a LaTeX special inside a TeX *text-mode* wrapper.

The docs site shipped ``global\\_opt`` — a visible backslash — in the rendered
equation on ``notebooks/amp_global_minlp.html``. The cause is a habit, not a
typo: in a LaTeX document ``\\text{global\\_opt}`` is exactly right, and MathJax 3
implements none of TeX's text-mode escapes, so it prints the backslash.

Measured 2026-09-24 against pdflatex 3.141592653 and MathJax 3 in Chromium
(24 compiles x 24 renderings, ``scratchpad/dual_probe.py``):

* inside ``\\text{}``/``\\texttt{}``  — ``\\_ \\% \\# \\&`` render literally under
  MathJax; the bare characters are a compile error under pdflatex. **No spelling
  satisfies both engines.**
* inside ``\\mathtt{}``/``\\mathrm{}`` or bare math — ``\\_ \\% \\# \\& \\$ \\{ \\}``
  are exact under both.

So the rule is mechanical: put an identifier that contains a special in a *math*
wrapper, never a *text* one. ``m.to_latex()`` is documented as markup to paste
into a paper, so "just render it for MathJax" is not available either — both
engines have to be right.

One grep is cheaper than another release that ships the artifact, which is what
this file is. The renderer half of the same rule is in
``test_display_cluster.py`` (``_latex_text`` / ``_MATH_ATOMS``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_DOCS = Path(__file__).resolve().parents[2] / "docs"

# A TeX text-mode wrapper and its (brace-free) contents.
_TEXT_WRAPPER = re.compile(r"\\(?:text|texttt|textbf|textit|textsf|mbox)\{([^{}]*)\}")

# What must not appear inside one: a backslash-escaped special, or one of the
# text-only glyph macros MathJax does not implement.
_TEXT_MODE_ESCAPE = re.compile(r"\\[_%#&~^]|\\text(?:backslash|asciitilde|asciicircum)")


def _markdown_sources() -> list[tuple[Path, str]]:
    """Every authored markdown source under docs/, notebooks included."""
    out: list[tuple[Path, str]] = []
    for path in sorted(_DOCS.rglob("*")):
        if "_build" in path.parts or path.suffix not in {".md", ".ipynb"}:
            continue
        if path.suffix == ".ipynb":
            nb = json.loads(path.read_text(encoding="utf-8"))
            for cell in nb.get("cells", []):
                if cell.get("cell_type") == "markdown":
                    out.append((path, "".join(cell.get("source", []))))
        else:
            out.append((path, path.read_text(encoding="utf-8")))
    return out


@pytest.mark.unit
def test_no_text_mode_escapes_in_authored_docs() -> None:
    sources = _markdown_sources()
    # Prove the probe fired: a rglob that matched nothing would report a clean
    # pass forever (CLAUDE.md, "Measurement & instrumentation discipline" §6).
    assert len(sources) > 50, f"only {len(sources)} sources scanned -- glob is wrong"

    scanned = 0
    offenders: list[str] = []
    for path, src in sources:
        for match in _TEXT_WRAPPER.finditer(src):
            scanned += 1
            if _TEXT_MODE_ESCAPE.search(match.group(1)):
                rel = path.relative_to(_DOCS.parent)
                offenders.append(f"{rel}: {match.group(0)}")
    assert scanned > 0, "no \\text{} wrappers found at all -- the regex is wrong"
    assert not offenders, (
        "MathJax renders TeX text-mode escapes literally; use a math wrapper "
        "(\\mathtt{...} / \\mathrm{...}) instead:\n  " + "\n  ".join(offenders)
    )
