#!/usr/bin/env python3
"""Build the retrieval index the in-browser docs assistant searches.

The assistant (``docs/_static/ask.js``) is retrieval-augmented: it looks up
passages in this index and hands them to a WebLLM model running in the
reader's own browser. The model never sees the corpus, only the passages a
question retrieves, so the index *is* the assistant's knowledge — a page
missing here is a page the assistant cannot answer about.

Why this reads **built HTML** and not the markdown source
--------------------------------------------------------

The equivalent tool in the POUNCE project parses its mdBook source, which is
plain markdown. That does not port. This book is Jupyter Book: two thirds of
its pages are ``.ipynb`` notebooks, and the ``.md`` pages carry MyST roles
(``{cite:p}``, ``{eval-rst}``, admonition directives) that are noise unless
rendered. More importantly, the *anchors* are not derivable from the source
without reimplementing Sphinx's slugifier — and an anchor that does not match
is a citation link that silently lands at the top of the page, which looks
like it worked.

Parsing ``docs/_build/html`` removes that whole class of bug: the anchors,
the page URLs and the rendered prose are read from the artifact the reader
will actually be served. The cost is an ordering constraint — build the book
first — which the ``docs`` Makefile target enforces.

Sphinx puts the ``id`` on the enclosing ``<section>``, not on the heading, so
chunking walks the section tree rather than scanning for headings. Sections
longer than MAX_CHARS are split at paragraph boundaries and each piece
re-carries the heading trail, so a passage is self-describing even when it is
the fourth slice of one section.

Usage:
    scripts/build-docs-index.py [-o OUT] [--html-dir DIR] [--with-autoapi]

Output is JSON on the schema documented in ``docs/ask.md``. Keys are short
(``t``/``h``/``u``/``k``/``x``) because this file is downloaded by every
reader who opens the assistant.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

# Passage sizing. MAX_CHARS is a retrieval choice, not a model-context one:
# short enough that a hit is specific, long enough that a worked example or a
# full option description survives in one piece. MIN_CHARS drops the
# heading-only stubs that would otherwise win on a title match and carry no
# answer.
MAX_CHARS = 1600
MIN_CHARS = 40

# A long code cell is bad context — it crowds out prose in the model's window
# and rarely contains the sentence that answers the question. Keep enough to
# show the shape of the call. Notebook pages make this load-bearing rather
# than cosmetic: a single solver-log output cell can run to hundreds of lines.
MAX_CODE_LINES = 25

# Pages that are navigation or machinery, never content.
SKIP_FILES = {
    "genindex.html",
    "search.html",
    "py-modindex.html",
    "searchindex.html",
}
SKIP_DIRS = {"_static", "_sources", "_images", "_downloads", "_sphinx_design_static"}

# The autoapi tree is machine-generated API reference: thousands of short
# signature stubs that would dominate the index by count and swamp the prose
# pages a question is usually about. Opt in with --with-autoapi.
AUTOAPI_DIR = "autoapi"

RE_WS = re.compile(r"[ \t]+")
RE_BLANKS = re.compile(r"\n{3,}")


def _pack(units: list[str], sep: str, max_chars: int) -> list[str]:
    """Greedily pack units into runs no longer than max_chars."""
    parts: list[str] = []
    buf = ""
    for unit in units:
        if not buf:
            buf = unit
        elif len(buf) + len(sep) + len(unit) <= max_chars:
            buf += sep + unit
        else:
            parts.append(buf)
            buf = unit
    if buf:
        parts.append(buf)
    return parts


def split_long(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split an over-long section into passages.

    Paragraph boundaries first, then a line-level pass for the paragraphs that
    are still too long on their own — a rendered markdown *table* is one
    paragraph, and the option tables in this book are long enough that a
    paragraph pass alone leaves chunks that would swamp a small model's whole
    context window with one hit.

    A single line longer than max_chars is still emitted whole: cutting
    mid-sentence costs more than the overrun.
    """
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    for para in _pack(text.split("\n\n"), "\n\n", max_chars):
        if len(para) <= max_chars:
            parts.append(para)
        else:
            parts.extend(_pack(para.split("\n"), "\n", max_chars))
    return parts


# Elements whose content is a single run of prose or code. Their inner markup
# is inline, so their text must be joined with *nothing* between the pieces.
LEAF_BLOCKS = ("p", "pre", "li", "dt", "dd", "td", "th", "figcaption", "caption")

# If one of the above contains one of these, it is not a leaf after all and is
# left alone so the nested block keeps its own boundary.
BLOCK_TAGS = ("p", "div", "pre", "ul", "ol", "dl", "table", "section", "blockquote")


def flatten_inline(article) -> None:
    """Collapse inline markup inside leaf blocks, in place.

    Pygments wraps *every* code token in its own ``<span>``, and Sphinx wraps
    every citation and cross-reference in an ``<a>``. Extracting such a block
    with a newline separator therefore explodes ``dm.Model("x")`` into eleven
    lines and ``Biegler [2010]`` into three — which is not what the page says,
    is not what a reader would copy, and is not what a small model should be
    shown as the source text.

    Flattening the leaf blocks first means the newline separator used later
    only ever falls *between* blocks, where it belongs. Any spacing that
    matters is already in the text nodes, so the pieces join with no separator.
    """
    for el in article.find_all(LEAF_BLOCKS):
        if el.find(BLOCK_TAGS) is not None:
            continue  # contains a nested block; not a leaf
        el.string = el.get_text("")


def clean_article(article) -> None:
    """Strip chrome and cap code blocks, in place.

    Everything removed here is either invisible to a reader (scripts, styles),
    pure navigation (permalinks, the copy-button furniture), or unindexable
    (images, MathJax duplicates of the same formula).
    """
    for sel in (
        "script",
        "style",
        "a.headerlink",
        "button",
        "div.admonition-title > a",
        "img",
        "mjx-container",
        "div.toctree-wrapper",
    ):
        for node in article.select(sel):
            node.decompose()

    flatten_inline(article)

    # Cap long <pre> blocks (code cells and, worse, solver-log output cells).
    for pre in article.select("pre"):
        lines = pre.get_text().split("\n")
        if len(lines) > MAX_CODE_LINES:
            kept = "\n".join(lines[:MAX_CODE_LINES])
            pre.string = f"{kept}\n… ({len(lines) - MAX_CODE_LINES} more lines)"


def section_text(sec) -> str:
    """Text of a section excluding its nested subsections and its own heading.

    Each subsection becomes its own chunk with its own anchor, so including
    them here would index the same prose several times and make the parent —
    which is longer — win BM25's length normalization against the specific
    child that actually answers.
    """
    parts: list[str] = []
    for child in sec.children:
        name = getattr(child, "name", None)
        if name == "section":
            continue
        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            continue
        text = child.get_text("\n", strip=True) if name else str(child).strip()
        if text:
            parts.append(text)
    body = "\n\n".join(parts)
    body = RE_WS.sub(" ", body)
    body = RE_BLANKS.sub("\n\n", body)
    return "\n".join(ln.rstrip() for ln in body.split("\n")).strip()


def heading_text(sec) -> str:
    h = sec.find(["h1", "h2", "h3", "h4", "h5", "h6"], recursive=False)
    if h is None:
        return ""
    return h.get_text(" ", strip=True).rstrip("#").strip()


def walk_sections(sec, url_base: str, title: str, trail: list[str], out: list[dict]) -> None:
    """Recursively emit one chunk group per section."""
    head = heading_text(sec)
    new_trail = trail + [head] if head else trail
    anchor = sec.get("id") or ""
    url = url_base + (f"#{anchor}" if anchor else "")
    body = section_text(sec)

    if body:
        heading = " › ".join(new_trail) if new_trail else title
        for piece in split_long(body):
            if len(piece) < MIN_CHARS:
                continue
            out.append({"t": title, "h": heading, "u": url, "k": "book", "x": piece})

    for child in sec.find_all("section", recursive=False):
        walk_sections(child, url_base, title, new_trail, out)


def chunks_for_page(html: str, url_base: str) -> tuple[list[dict], str]:
    """Turn one built page into indexable chunks."""
    soup = BeautifulSoup(html, "html.parser")
    article = soup.select_one("article.bd-article") or soup.select_one("article")
    if article is None:
        return [], ""

    clean_article(article)

    h1 = article.find("h1")
    title = h1.get_text(" ", strip=True).rstrip("#").strip() if h1 else url_base
    if not title:
        title = url_base

    out: list[dict] = []
    top = article.find_all("section", recursive=False)
    if top:
        for sec in top:
            walk_sections(sec, url_base, title, [], out)
    else:
        # A page with no <section> wrapper (rare, e.g. a bare directive page).
        body = section_text(article)
        for piece in split_long(body):
            if len(piece) >= MIN_CHARS:
                out.append({"t": title, "h": title, "u": url_base, "k": "book", "x": piece})
    return out, title


def iter_pages(html_dir: Path, with_autoapi: bool):
    for path in sorted(html_dir.rglob("*.html")):
        rel = path.relative_to(html_dir)
        if rel.name in SKIP_FILES:
            continue
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if not with_autoapi and rel.parts and rel.parts[0] == AUTOAPI_DIR:
            continue
        yield path, rel.as_posix()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-o",
        "--out",
        default="docs/_build/html/_static/ask-index.json",
        help="output JSON path (default: into the built site, next to ask.js)",
    )
    ap.add_argument(
        "--html-dir",
        default="docs/_build/html",
        help="built Jupyter Book HTML root",
    )
    ap.add_argument(
        "--with-autoapi",
        action="store_true",
        help="also index the generated autoapi/ API reference (large)",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    html_dir = Path(args.html_dir)
    if not html_dir.is_dir():
        print(
            f"build-docs-index: no built book at {html_dir} — run `make docs` first",
            file=sys.stderr,
        )
        return 1

    chunks: list[dict] = []
    pages = 0
    empty_pages: list[str] = []
    for path, rel in iter_pages(html_dir, args.with_autoapi):
        page_chunks, _title = chunks_for_page(path.read_text(encoding="utf-8"), rel)
        pages += 1
        if not page_chunks:
            empty_pages.append(rel)
        chunks.extend(page_chunks)

    # Prove the probe fired. An extractor that silently matches nothing (a
    # changed theme class, an empty build) would otherwise ship a valid-looking
    # index that answers every question with "no passages found".
    if pages == 0:
        print(f"build-docs-index: no pages found under {html_dir}", file=sys.stderr)
        return 1
    if not chunks:
        print(
            f"build-docs-index: {pages} pages yielded 0 chunks — the content "
            "selector matched nothing (did the theme change?)",
            file=sys.stderr,
        )
        return 1

    anchored = sum(1 for c in chunks if "#" in c["u"])
    if anchored == 0:
        print(
            "build-docs-index: no chunk carries an anchor — every citation "
            "would land at the top of its page",
            file=sys.stderr,
        )
        return 1

    for i, c in enumerate(chunks):
        c["i"] = i

    doc = {
        "schema": 1,
        "counts": {"pages": pages, "total": len(chunks), "anchored": anchored},
        "chunks": chunks,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")

    if not args.quiet:
        size = os.path.getsize(out_path)
        print(
            "build-docs-index: %d chunks from %d pages (%d anchored, %.0f%%) "
            "-> %s (%.0f KB)"
            % (
                len(chunks),
                pages,
                anchored,
                100.0 * anchored / len(chunks),
                out_path,
                size / 1024.0,
            )
        )
        if empty_pages:
            print(
                "build-docs-index: %d page(s) produced no passages: %s"
                % (len(empty_pages), ", ".join(empty_pages[:8]))
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
