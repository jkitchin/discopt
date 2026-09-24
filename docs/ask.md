# Ask discopt: the in-browser docs assistant

The floating **Ask** button in the lower-right corner of every page opens a
question box for these docs. It has two halves, and they work independently:

| | What it does | What it costs |
|---|---|---|
| **Search** | Ranks passages from this book, deep-linked to the exact heading | one ~1.3 MB index download, on first use |
| **Answer** | Writes those passages up into prose, with citations | a language model download, hundreds of MB, only if you ask for it |

Search always works. The answer half is strictly opt-in: nothing is downloaded
until you pick a model and click **Load model**. If you never do, you still get
ranked, linked passages, which is most of the value on a reference manual.

Nothing you type leaves your browser. There is no API key, no server, and no
telemetry — the assistant is a static JavaScript file, and the only
third-party request it ever makes is fetching model weights from the WebLLM
CDN after you click.

This is not the same thing as {doc}`discopt's own LLM features
<llm_features>`. Those call a hosted model through litellm, need an API key,
and can see your actual model; this one runs in your browser, needs nothing,
and can only see these docs.

## What it searches

Every page of this book — the tutorials, the modeling guide, the solver
internals, the comparison and semantics pages. Two thirds of the corpus is
notebooks, so a question about how to *do* something usually lands on a
worked example rather than on a prose description of one.

The generated API reference (`autoapi/`) is deliberately **out**. It is
thousands of short signature stubs, and indexing it would swamp the prose
pages a question is normally about — a hundred one-line entries that each
mention `Model` outranking the modeling guide's section on it. Use the
{doc}`API reference <api>` directly, or the book's own search box, for
"what arguments does this take".

Passages are cut at heading boundaries, so every citation lands on the section
that answered rather than at the top of a long notebook.

## Requirements for written answers

| | |
|---|---|
| **Browser** | WebGPU: Chrome/Edge 113+, Safari 26+, Firefox with WebGPU enabled |
| **Download** | 0.9–2.3 GB depending on the model, cached by the browser afterwards |
| **Memory** | roughly the model's size in GPU memory while loaded |

Without WebGPU the panel says so and stays in search-only mode.

Three models are offered, smallest first — Llama 3.2 1B, Qwen 2.5 1.5B, and
Llama 3.2 3B. The smallest is the default because a 0.9 GB download that
answers beats a 2.3 GB one you abandon; the larger ones follow instructions
and citation formatting more reliably. If your GPU reports `shader-f16` you
get the half-precision builds, which are about a third smaller.

## How much to trust it

The model only ever sees the five passages your question retrieved, and it is
told to answer from them alone and to cite each claim. That keeps it far
closer to the docs than an unaided chatbot — but a 1B model reading correct
passages can still summarize them wrongly.

**Treat the answer as a routing device and the passages as the source.** Every
answer is shown with the passages it was built from, in rank order, linked to
the heading they came from. When the two disagree, the passage is right.

The retrieval half is worth stating plainly too: it is
[BM25](https://en.wikipedia.org/wiki/Okapi_BM25) over words, not embeddings.
It is excellent at exact names — an API call copied straight out of your
script, or a solver status string copied out of your log — and weaker at
questions phrased with no term in common with the docs. If a question comes
back empty, search for the function name or the exact status string.

Measured over a 20-query labelled set (`docs/tests/ask_retrieval.mjs`), the
right page is the top hit for 18 and in the top five for all 20.

## For maintainers

| Piece | File |
|---|---|
| Index builder | `scripts/build-docs-index.py` |
| Panel, retrieval, WebLLM glue | `docs/_static/ask.js`, `docs/_static/ask.css` |
| Retrieval guard | `docs/tests/ask_retrieval.mjs` (ranking, on the index alone) |
| Browser guard | `docs/tests/ask_e2e.py` (the panel and its anchors, in Chromium) |
| Wiring | `html_js_files` in `docs/_config.yml` |

`make ask-check` runs both guards against the current build.

Three details that are easy to get wrong:

- **The index is built from the rendered HTML, not from the source.** The
  equivalent tool in the POUNCE project, which this is ported from, parses
  mdBook markdown. That does not port: two thirds of these pages are
  notebooks, the `.md` pages carry MyST roles that are noise unless rendered,
  and — the real reason — the heading anchors are not derivable from the
  source without reimplementing Sphinx's slugifier. An anchor that does not
  match is a citation that silently lands at the top of the page, which looks
  like it worked. Reading `docs/_build/html` takes the anchors, the URLs and
  the prose from the artifact the reader is actually served.

- **The index is therefore a post-build step, and `make docs` runs both.**
  A bare `jupyter-book build docs/` produces a book whose assistant searches
  nothing: `ask-index.json` is not a source file and is not in `_static/`
  until the builder writes it there. The panel reports the failed fetch rather
  than pretending, but the way not to see that is to use `make docs`.

- **Code must survive extraction as code.** Pygments wraps every token in its
  own `<span>`. Joining those with a newline — the obvious way to get block
  structure out of HTML — turns `dm.Model("x")` into eleven lines, and the
  index still looks fine and still returns hits, just never for the name a
  reader typed. `flatten_inline()` in the builder exists for that, and
  `ask_retrieval.mjs` asserts `dm.Model` and `subject_to` survive whole.

The index itself is one JSON file of passages. Keys are one character because
every reader who opens the panel downloads it:

```json
{"schema": 1,
 "counts": {"pages": 87, "total": 1119, "anchored": 1119},
 "chunks": [{"t": "Modeling guide",
             "h": "Modeling guide > Variables",
             "u": "notebooks/modeling_guide.html#variables",
             "k": "book",
             "x": "A variable is created with ...",
             "i": 0}]}
```

`t` page title, `h` the heading trail shown under a citation, `u` the URL
relative to the site root, `k` the corpus (only `book` today), `x` the passage
text, `i` its position within the page. `counts.anchored` is the number of
chunks whose `u` carries a `#fragment`; the builder exits non-zero if it is
zero, along with zero pages and zero chunks, so a run that quietly indexed
nothing cannot report success.

To build and check the assistant locally:

```console
$ make docs                                   # builds the book, then the index
$ make ask-check                              # rebuilds the index and guards it
$ python3 -m http.server -d docs/_build/html 8000
```

Serve it rather than opening `file://`: the index is fetched with `fetch()`,
which most browsers refuse on a `file://` origin. (`ask_e2e.py` serves the
build itself, so it needs no server of your own — just
`pip install playwright && playwright install chromium`.)
