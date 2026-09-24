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
