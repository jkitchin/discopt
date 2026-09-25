#!/usr/bin/env python3
"""End-to-end check of the docs assistant in a real browser.

``ask_retrieval.mjs`` next door tests the ranking in isolation, on the index
alone. This tests the half that only exists once a browser runs the page: that
the panel mounts, that ask.js locates its own stylesheet and index relative to
its ``<script src>`` rather than to the current page, and -- the reason the
index is built from rendered HTML at all -- that the anchors it cites are real.
A citation whose fragment does not match silently lands at the top of the page
and looks like it worked, so this follows every citation and asserts a mid-page
one actually scrolls.

It drives a NESTED page deliberately: a static root resolved against the page
instead of against the script works on ``/index.html`` and 404s everywhere else.

Serves the build itself and picks its own port, so it needs no setup beyond
``make docs`` and ``pip install playwright && playwright install chromium``.

Prints an executed-check count and exits non-zero if it is zero, so a run that
silently asserted nothing cannot read as a pass.

Usage:
    python docs/tests/ask_e2e.py [--html-dir docs/_build/html]
"""

import argparse
import contextlib
import functools
import pathlib
import socket
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--html-dir", default="docs/_build/html", help="the built book")
args = parser.parse_args()

if not (pathlib.Path(args.html_dir) / "index.html").exists():
    sys.exit(f"ask_e2e: no build at {args.html_dir} -- run `make docs` first")


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *a):  # noqa: D102 - silence per-request logging
        pass


with socket.socket() as probe:
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]

server = ThreadingHTTPServer(
    ("127.0.0.1", port), functools.partial(_Quiet, directory=args.html_dir)
)
threading.Thread(target=server.serve_forever, daemon=True).start()

BASE = f"http://127.0.0.1:{port}"
NESTED = f"{BASE}/notebooks/bound_tightening.html"

checks = 0
fails = []


def check(label, cond, detail=""):
    global checks
    checks += 1
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label} {detail}")
        fails.append(label)


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()

    console_errors = []
    page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: console_errors.append(str(e)))

    requests = []
    page.on("response", lambda r: requests.append((r.url, r.status)))

    page.goto(NESTED, wait_until="networkidle")

    # The stylesheet ask.js links itself, relative to its own <script src>.
    css = [u for u, _ in requests if u.endswith("ask.css")]
    check("ask.css fetched from a nested page", bool(css), css)
    check(
        "ask.css resolved under _static/",
        bool(css) and "/_static/ask.css" in css[0],
        css[0] if css else "",
    )
    check(
        "ask.css served 200",
        any(s == 200 for u, s in requests if u.endswith("ask.css")),
        [(u, s) for u, s in requests if u.endswith("ask.css")],
    )

    toggle = page.locator(".discopt-ask-toggle")
    check("Ask pill is present", toggle.count() == 1, toggle.count())
    check("Ask pill is visible", toggle.count() == 1 and toggle.is_visible())

    toggle.click()
    check("panel opens", page.locator(".discopt-ask-panel").is_visible())

    # ---- The open panel makes room; it does not cover the page. ------------
    # Before this was fixed the panel was a plain fixed overlay at right: 0,
    # and on the default 1280px viewport it hid 208px of <article> and the
    # whole 272px secondary sidebar -- which is what a reader sees as the
    # "Contents" list sliced off down its right edge. Geometry, not a
    # screenshot, so it cannot pass by rendering differently.
    def _overlap(a, b):
        return min(a["x"] + a["width"], b["x"] + b["width"]) - max(a["x"], b["x"])

    panel_box = page.locator(".discopt-ask-panel").bounding_box()
    measured = 0
    for sel in ("article.bd-article", ".bd-sidebar-secondary", ".bd-sidebar-primary"):
        loc = page.locator(sel)
        if loc.count() == 0 or not loc.first.is_visible():
            continue
        bb = loc.first.bounding_box()
        if bb is None or bb["width"] == 0:
            continue
        measured += 1
        ov = _overlap(bb, panel_box)
        # 1px of slack for subpixel layout, not for a real overlap.
        check(f"open panel does not cover {sel}", ov <= 1, f"{ov:.0f}px covered")
    # Without this the loop above is a no-op that reports nothing and reads as
    # a pass -- the exact shape CLAUDE.md section 6 exists to forbid.
    check("occlusion was actually measured against page content", measured >= 2, measured)

    check(
        "reflow introduced no horizontal scrollbar",
        not page.evaluate(
            "document.documentElement.scrollWidth > document.documentElement.clientWidth"
        ),
    )

    # ---- The width is draggable, clamped, and remembered. ------------------
    handle = page.locator(".discopt-ask-resize")
    check("resize handle is present", handle.count() == 1, handle.count())
    if handle.count() == 1:
        before = panel_box["width"]
        hb = handle.bounding_box()
        page.mouse.move(hb["x"] + hb["width"] / 2, hb["y"] + 200)
        page.mouse.down()
        page.mouse.move(hb["x"] - 120, hb["y"] + 200, steps=8)
        page.mouse.up()
        widened = page.locator(".discopt-ask-panel").bounding_box()["width"]
        check("dragging the handle widens the panel", widened > before + 100, (before, widened))

        art = page.locator("article.bd-article").first.bounding_box()
        check(
            "the widened panel still covers nothing",
            _overlap(art, page.locator(".discopt-ask-panel").bounding_box()) <= 1,
        )

        # A drag past the floor must stop at it rather than collapse the panel.
        hb = handle.bounding_box()
        page.mouse.move(hb["x"] + hb["width"] / 2, hb["y"] + 200)
        page.mouse.down()
        page.mouse.move(page.viewport_size["width"] + 400, hb["y"] + 200, steps=8)
        page.mouse.up()
        floored = page.locator(".discopt-ask-panel").bounding_box()["width"]
        check("a drag past the minimum clamps instead of collapsing", floored >= 279, floored)

        stored = page.evaluate("localStorage.getItem('discopt-ask-width')")
        check("the dragged width is remembered", stored is not None and stored.isdigit(), stored)

    # Closing gives the page its width back.
    page.locator(".discopt-ask-close").click()
    check(
        "closing restores the body padding",
        page.evaluate("getComputedStyle(document.body).paddingRight") in ("0px", ""),
        page.evaluate("getComputedStyle(document.body).paddingRight"),
    )
    toggle.click()

    box = page.locator(".discopt-ask-input")
    box.fill("feasibility based bound tightening")
    page.locator(".discopt-ask-submit").click()

    # The index is fetched lazily on the first question.
    page.wait_for_selector(".discopt-ask-source-link", timeout=30000)

    idx = [(u, s) for u, s in requests if u.endswith("ask-index.json")]
    check("index fetched", bool(idx), idx)
    check(
        "index resolved under _static/",
        bool(idx) and "/_static/ask-index.json" in idx[0][0],
        idx[0][0] if idx else "",
    )
    check("index served 200", bool(idx) and idx[0][1] == 200, idx)

    links = page.locator(".discopt-ask-source-link")
    n = links.count()
    check("sources returned", n > 0, n)

    href = links.first.get_attribute("href")
    check("top citation is absolute http(s)", bool(href) and href.startswith("http"), href)
    check("top citation carries an anchor", bool(href) and "#" in href, href)

    # Footer "How this works" must reach the page we wrote.
    doc_href = page.locator(".discopt-ask-foot a").first.get_attribute("href")
    check("footer links ask.html", bool(doc_href) and doc_href.endswith("/ask.html"), doc_href)

    # Follow every citation: each anchor must exist on the page it cites.
    hrefs = [links.nth(i).get_attribute("href") for i in range(n)]
    for h in hrefs:
        frag = h.split("#", 1)[1]
        page.goto(h, wait_until="networkidle")
        check(f"cited anchor exists: {frag}", page.locator(f'[id="{frag}"]').count() > 0)

    # A citation into the middle of a page must actually scroll there -- the
    # silent-failure mode this whole design exists to avoid is an anchor that
    # does not match and quietly lands at the top. (The first section of a page
    # legitimately sits at scrollY 0, so test one that is not first.)
    deep = None
    for h in hrefs:
        page.goto(h, wait_until="networkidle")
        frag = h.split("#", 1)[1]
        first = page.evaluate("(document.querySelector('article.bd-article section')||{}).id")
        if frag != first:
            deep = (h, frag, page.evaluate("window.scrollY"))
            break
    check("a mid-page citation was available to test", deep is not None)
    if deep:
        check(f"mid-page citation scrolls to {deep[1]}", deep[2] > 0, deep[2])

    # ---- The answer renders as prose, not as its own source. --------------
    # A reader reported seeing a literal "\\[ \\min_x ... \\]" where the formula
    # belongs and "[2]" as three inert characters beside the passage list it
    # names. renderAnswer now parses markdown and TeX; this drives the SHIPPED
    # renderer through the seam ask.js exposes, with real hits already loaded
    # from the search above, so the citation links resolve against real chunks.
    page.goto(NESTED, wait_until="networkidle")
    page.locator(".discopt-ask-toggle").click()
    page.locator(".discopt-ask-input").fill("feasibility based bound tightening")
    page.locator(".discopt-ask-submit").click()
    page.wait_for_selector(".discopt-ask-source-link", timeout=30000)

    seam = page.evaluate("!!document.querySelector('.discopt-ask-panel').__discoptRenderAnswer")
    check("the renderer seam is exposed", seam)

    ANSWER = (
        "## Quadratic programming\n\n"
        "QP has the form\n\n"
        "\\[\n\\min_{x} \\tfrac{1}{2} x^\\top Q x\n\\]\n\n"
        "where \\(Q\\) is symmetric [1]. Build one with `dm.Model` [2].\n\n"
        "- **Convex** when Q is PSD [1]\n\n"
        "```python\nm = dm.Model()\n```\n"
    )
    page.evaluate(
        "(t) => document.querySelector('.discopt-ask-panel').__discoptRenderAnswer(t, true)",
        ANSWER,
    )
    page.wait_for_timeout(1500)  # MathJax typesets asynchronously

    answer = page.locator(".discopt-ask-answer")
    check("the answer heading is an element", answer.locator("h3, h4, h5").count() > 0)
    check("a fenced block becomes <pre><code>", answer.locator("pre code").count() > 0)
    check("a list item becomes <li>", answer.locator("li").count() > 0)
    check("bold becomes <strong>", answer.locator("strong").count() > 0)

    # Citations: links, pointing at the passages the sources list shows.
    cites = answer.locator("a.discopt-ask-cite")
    check("citations render as links", cites.count() >= 2, cites.count())
    cite_hrefs = [cites.nth(i).get_attribute("href") for i in range(cites.count())]
    check(
        "every citation link carries an anchor into the book",
        all(h and h.startswith("http") and "#" in h for h in cite_hrefs),
        cite_hrefs[:3],
    )
    src_hrefs = [
        page.locator(".discopt-ask-source-link").nth(i).get_attribute("href")
        for i in range(page.locator(".discopt-ask-source-link").count())
    ]
    check(
        "[1] points at the first listed passage",
        bool(cite_hrefs) and bool(src_hrefs) and cite_hrefs[0] == src_hrefs[0],
        (cite_hrefs[0] if cite_hrefs else None, src_hrefs[0] if src_hrefs else None),
    )
    check("citation text still reads [n]", cites.first.inner_text().strip() == "[1]")

    # MathJax: the integration node cannot test. The book already loads
    # MathJax 3, and it reads textContent -- so the no-HTML invariant holds and
    # a typeset display block leaves an <mjx-container> behind.
    check(
        "MathJax 3 is available to the panel",
        page.evaluate("!!(window.MathJax && window.MathJax.typesetPromise)"),
    )
    check(
        "display math is typeset, not shown as raw TeX",
        answer.locator("mjx-container").count() > 0,
        answer.locator("mjx-container").count(),
    )
    check(
        "no raw TeX delimiter survives in the answer",
        "\\[" not in answer.inner_text(),
        answer.inner_text()[:120],
    )

    # Malformed TeX from a small local model. Reported from the live panel: a
    # 1B model wrote \begin{split} and closed it with \end{aligned}, and MathJax
    # drew its red-on-yellow "\begin{split} ended with \end{aligned}" box into
    # the middle of the answer. typesetPromise RESOLVES on a TeX error, so the
    # .catch never fires -- the check has to be for <mjx-merror> in the output.
    BAD = (
        "1. The standard form of a Linear Program is given by:\n\n"
        "\\[\n\\begin{split}\n\\min_x c^\\top x \\\\ Ax = b\n\\end{aligned}\n\\]\n\n"
        "and inline \\(\\begin{split} x \\end{aligned}\\) too.\n"
    )
    page.evaluate(
        "(t) => document.querySelector('.discopt-ask-panel').__discoptRenderAnswer(t, true)",
        BAD,
    )
    page.wait_for_timeout(1500)
    check(
        "a MathJax error box never reaches the reader",
        answer.locator("mjx-merror").count() == 0,
        answer.locator("mjx-merror").count(),
    )
    check(
        "the rejected expression is demoted to prose",
        answer.locator(".discopt-ask-math-failed").count() >= 1,
        answer.locator(".discopt-ask-math-failed").count(),
    )
    bad_text = answer.inner_text()
    check(
        "the demoted expression is readable, not raw TeX",
        "\\begin" not in bad_text and "ended with" not in bad_text,
        bad_text[:160],
    )
    check("the prose around the bad math survives", "standard form" in bad_text)

    # The invariant the renderer exists to protect: model text is never markup.
    page.evaluate(
        "(t) => document.querySelector('.discopt-ask-panel').__discoptRenderAnswer(t, true)",
        "<img src=x onerror=alert(1)> and [evil](javascript:alert(2))",
    )
    check("model output cannot inject an element", answer.locator("img").count() == 0)
    check(
        "a javascript: link is rendered as text, not a link",
        answer.locator("a[href^='javascript']").count() == 0,
    )
    check("the escaped markup is still shown to the reader", "<img" in answer.inner_text())

    # The docs page itself renders.
    page.goto(f"{BASE}/ask.html", wait_until="networkidle")
    check(
        "ask.html has the expected title", "in-browser docs assistant" in page.title(), page.title()
    )

    # Scoped to the assistant: Jupyter Book itself logs a duplicate
    # THEBE_JS_URL declaration and a theme-mode complaint on every page, with
    # ask.js blocked as well as loaded (verified), so a blanket "no console
    # errors" here would assert someone else's bug and never ours.
    ours = [e for e in console_errors if "ask" in e.lower() or "discopt-ask" in e.lower()]
    check("no console errors from the assistant", not ours, ours[:3])

    browser.close()

with contextlib.suppress(Exception):
    server.shutdown()

print(f"\nask_e2e: {checks} checks, {len(fails)} failed")
if checks == 0:
    print("ask_e2e: NO CHECKS RAN")
    sys.exit(2)
sys.exit(1 if fails else 0)
