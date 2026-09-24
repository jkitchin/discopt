// discopt docs assistant — retrieval-augmented, entirely in the reader's browser.
//
// Ported from the POUNCE project's assistant. Two halves that are deliberately
// independent:
//
//   RETRIEVAL  BM25 over ask-index.json (built by scripts/build-docs-index.py
//              from the BUILT book under docs/_build/html). Pure JS, no model,
//              no network beyond the index itself. This half always works.
//
//   GENERATION WebLLM (https://github.com/mlc-ai/web-llm) running a small
//              instruct model on WebGPU, fed *only* the retrieved passages.
//              Strictly opt-in: the weights are hundreds of megabytes, so
//              nothing is fetched until the reader clicks "Load model".
//
// The split is the point. A reader without WebGPU, without the patience for a
// model download, or on a phone still gets ranked deep-linked passages — which
// is most of the value on a reference manual — and a reader who loads the
// model gets those same passages written up. Nothing degrades to a blank panel.
//
// Nothing leaves the browser. There is no API key, no server, no telemetry;
// the only third-party traffic is the model weight download from the WebLLM
// CDN, and that happens only after the explicit click.
//
// Delivery is Sphinx's ordinary html_js_files / html_css_files (see
// docs/_config.yml), so this file is served from _static/ alongside ask.css
// and ask-index.json. It locates all three from its own <script src> rather
// than from a configured base path, so the book works served at any path or
// domain — including from a file:// checkout or a versioned subdirectory.
(function () {
  var SELF_ID = "discopt-ask";
  var WEBLLM_URL = "https://esm.run/@mlc-ai/web-llm@0.2.84";

  // Offered smallest-first: the first entry is what an undecided reader gets,
  // and a 900 MB download that answers is worth more than a 2.3 GB one they
  // abandon. `vram` is the MB figure from WebLLM's own prebuiltAppConfig.
  var MODELS = [
    { base: "Llama-3.2-1B-Instruct", label: "Llama 3.2 1B", vram: 879 },
    { base: "Qwen2.5-1.5B-Instruct", label: "Qwen 2.5 1.5B", vram: 1630 },
    { base: "Llama-3.2-3B-Instruct", label: "Llama 3.2 3B", vram: 2264 }
  ];

  var TOP_K = 5; // passages handed to the model
  var TOP_SHOW = 6; // passages listed as sources
  var MAX_CTX_CHARS = 900; // per passage, into the prompt
  var LS_MODEL = "discopt-ask-model";

  var state = {
    index: null, // { chunks, N, avgdl, df, docs }
    loading: null, // in-flight index fetch
    engine: null, // WebLLM engine once loaded
    engineModel: null,
    busy: false,
    lastHits: []
  };

  // ---------------------------------------------------------------- paths --

  // Captured at load, not looked up later: `document.currentScript` is only
  // non-null while this file is executing, and every caller below runs from an
  // event handler or from DOMContentLoaded, by which time it is null. Getting
  // this wrong does not throw — it silently resolves every path against the
  // current page instead of against _static/, so the index 404s on any page
  // that is not at the book root.
  var SELF_SCRIPT = typeof document !== "undefined" ? document.currentScript : null;

  function self() {
    return (
      (typeof document !== "undefined" && document.getElementById(SELF_ID)) ||
      SELF_SCRIPT ||
      (typeof document !== "undefined"
        ? document.querySelector('script[src$="ask.js"]')
        : null)
    );
  }

  function attr(name, fallback) {
    var el = self();
    var v = el && el.getAttribute && el.getAttribute(name);
    return v === null || v === undefined ? fallback : v;
  }

  // Everything this file needs — the stylesheet, the index, and the "how this
  // works" page — is located from its own <script src>, which Sphinx emits as
  // an absolute path to _static/ask.js on every page at every nesting depth.
  //
  // Deriving the paths beats configuring them. A relative base injected per
  // page is only ever wrong at a depth nobody tested, and the failure is
  // silent: a 404 on the index reads to the reader as "the assistant is
  // broken" with nothing naming the cause.
  function staticRoot() {
    var el = self();
    var src = (el && el.src) || "";
    return new URL(".", src || window.location.href);
  }

  // The book root is the parent of _static/.
  function siteRoot() {
    return new URL("../", staticRoot());
  }

  function indexUrl() {
    return new URL(attr("data-index", "ask-index.json"), staticRoot());
  }

  // A chunk's `u` is a path relative to the book root — never to the current
  // page, which is why this resolves against siteRoot() rather than against
  // window.location.
  function hrefFor(chunk) {
    if (/^https?:/.test(chunk.u)) return chunk.u;
    try {
      return new URL(chunk.u, siteRoot()).href;
    } catch (e) {
      return chunk.u;
    }
  }

  // ------------------------------------------------------------ retrieval --

  // Function words carry no signal and actively mislead here. A question
  // phrased as a question ("what does MaximumIterationsExceeded mean", "my
  // solve fails because of where it started") is mostly stopwords, and
  // without this list the short passages that happen to contain several of
  // them outrank the passage that contains the one term that matters.
  // Ablated against the final system on the 20-query eval set, this list is
  // worth top-1 17/20 -> 14/20 (book-only).
  //
  // Deliberately *not* included: "no", "not", "off", "on", "up", "down",
  // "over", "under" — each is load-bearing somewhere in this corpus
  // ("turning it off", "warm start up", "bound tightening").
  var STOPWORDS = (
    "a about all also am an and any are as at be because been but by can " +
    "could did do does doing done for from get give go had has have he her " +
    "here him his how i if in into is it its just like make me might more " +
    "most much must my need of one only or other our out own please should " +
    "since so some such than that the their them then there these they this " +
    "those through to too us use used using very want was way we were what " +
    "when where whether which while who why will with would you your"
  ).split(" ").reduce(function (set, w) {
    set[w] = 1;
    return set;
  }, Object.create(null));

  // Porter step 1 — inflectional endings only (plurals, -ed, -ing, -y).
  //
  // Steps 2-5 are derivational (-ational -> -ate, -iveness -> -ive) and are
  // left out: they buy little on a corpus whose distinctive vocabulary is
  // identifiers and proper nouns, and every extra rule is another chance to
  // collide two terms that a solver manual distinguishes.
  //
  // Step 1 alone is what a reader's phrasing needs. Ablated to an identity
  // stem, top-1 falls 17/20 -> 14/20 (book-only); it is what connects "why does *relaxing* the bounds…" to the passages
  // about bound *relaxation*, and "my solve fails because of where it
  // *started*" to "*starting* point". The +e restoration in 1b matters as
  // much as the stripping: without it "scaling" stems to `scal` and no longer
  // matches "scale".
  var CONS = /[^aeiou]/;

  function isCons(w, i) {
    var c = w.charAt(i);
    if (c === "y") return i === 0 ? true : !isCons(w, i - 1);
    return CONS.test(c);
  }

  function hasVowel(w) {
    for (var i = 0; i < w.length; i++) {
      if (!isCons(w, i)) return true;
    }
    return false;
  }

  // Porter's `m`: the number of vowel-consonant sequences in the stem.
  function measure(w) {
    var n = 0;
    var i = 0;
    while (i < w.length && isCons(w, i)) i++;
    while (i < w.length) {
      while (i < w.length && !isCons(w, i)) i++;
      if (i >= w.length) break;
      n++;
      while (i < w.length && isCons(w, i)) i++;
    }
    return n;
  }

  function endsDoubleCons(w) {
    var n = w.length;
    return n > 1 && w.charAt(n - 1) === w.charAt(n - 2) && isCons(w, n - 1);
  }

  // consonant-vowel-consonant where the final consonant is not w, x or y.
  function cvc(w) {
    var n = w.length;
    if (n < 3) return false;
    if (!isCons(w, n - 1) || isCons(w, n - 2) || !isCons(w, n - 3)) return false;
    return "wxy".indexOf(w.charAt(n - 1)) < 0;
  }

  function restore(stem) {
    if (/(at|bl|iz)$/.test(stem)) return stem + "e";
    if (endsDoubleCons(stem) && !/[lsz]$/.test(stem)) return stem.slice(0, -1);
    if (measure(stem) === 1 && cvc(stem)) return stem + "e";
    return stem;
  }

  var stemCache = Object.create(null);

  function normTok(t) {
    // Identifiers keep their exact form: `bound_relax_factor` and
    // `theta_max` are names, not English, and stemming their parts would
    // make an exact-name lookup fuzzy in precisely the case where the
    // reader was being precise.
    if (t.length < 4 || t.indexOf("_") >= 0 || /\d/.test(t)) return t;

    var hit = stemCache[t];
    if (hit !== undefined) return hit;

    var w = t;
    // 1a — plurals.
    if (/sses$/.test(w)) w = w.slice(0, -2);
    else if (/ies$/.test(w)) w = w.slice(0, -3) + "i";
    else if (/[^s]s$/.test(w)) w = w.slice(0, -1);

    // 1b — -eed / -ed / -ing.
    if (/eed$/.test(w)) {
      if (measure(w.slice(0, -1)) > 0) w = w.slice(0, -1);
    } else if (/ed$/.test(w) && hasVowel(w.slice(0, -2))) {
      w = restore(w.slice(0, -2));
    } else if (/ing$/.test(w) && hasVowel(w.slice(0, -3))) {
      w = restore(w.slice(0, -3));
    }

    // 1c — terminal y after a vowel-containing stem.
    if (/y$/.test(w) && hasVowel(w.slice(0, -1))) w = w.slice(0, -1) + "i";

    stemCache[t] = w;
    return w;
  }

  // Snake_case identifiers are emitted whole *and* split, so "bound relax
  // factor" and "bound_relax_factor" both hit the same passages.
  function tokenize(s) {
    var raw = String(s).toLowerCase().match(/[a-z0-9_]+/g) || [];
    var out = [];
    for (var i = 0; i < raw.length; i++) {
      var t = raw[i];
      if (STOPWORDS[t]) continue;
      out.push(normTok(t));
      if (t.indexOf("_") > 0) {
        var parts = t.split("_");
        for (var j = 0; j < parts.length; j++) {
          if (parts[j] && !STOPWORDS[parts[j]]) out.push(normTok(parts[j]));
        }
      }
    }
    return out;
  }

  function buildIndex(chunks) {
    var df = Object.create(null);
    var docs = new Array(chunks.length);
    var total = 0;

    for (var i = 0; i < chunks.length; i++) {
      var c = chunks[i];
      // The full trail goes into the searchable text — an ancestor heading is
      // real context for recall. But the *bonus* below is keyed on the leaf
      // heading and the page title only. Bonusing every ancestor token gives
      // deeply nested sections a boost proportional to their depth, which is
      // not a relevance signal: measured, it put
      // "Active-Set SQP Warm Starts › … › 1.2 Design decisions" above the
      // wiki page literally named "Recovering from a bad start".
      var trail = c.h.split("›");
      var leafTokens = tokenize(trail[trail.length - 1]);
      var titleTokens = tokenize(c.t);
      var tokens = tokenize(c.h).concat(titleTokens, tokenize(c.x));

      var tf = Object.create(null);
      for (var j = 0; j < tokens.length; j++) {
        tf[tokens[j]] = (tf[tokens[j]] || 0) + 1;
      }
      var head = Object.create(null);
      for (var k = 0; k < leafTokens.length; k++) head[leafTokens[k]] = 1;
      var title = Object.create(null);
      for (var m = 0; m < titleTokens.length; m++) title[titleTokens[m]] = 1;

      for (var term in tf) df[term] = (df[term] || 0) + 1;
      docs[i] = { tf: tf, len: tokens.length, head: head, title: title };
      total += tokens.length;
    }

    return {
      chunks: chunks,
      docs: docs,
      df: df,
      N: chunks.length,
      avgdl: chunks.length ? total / chunks.length : 1
    };
  }

  // Okapi BM25 with the usual constants, plus a flat bonus for a query term
  // that appears in the passage's heading trail — on a reference manual the
  // heading is very often the whole question ("what does scaling do").
  var K1 = 1.2;
  var B = 0.75;
  var HEAD_BONUS = 0.6;
  // The page title is the coarsest and most reliable topic signal in the
  // corpus: a reader asking about scaling wants the page called "Scaling"
  // before a subsection of "Sensitivity Analysis" that mentions it. Ablated,
  // this and HEAD_BONUS together are the largest single contributor to
  // ranking quality: dropping both takes top-1 17/20 -> 13/20 (book-only).
  var TITLE_BONUS = 0.5;

  // Query-side terms, deduplicated (a word repeated in the question must not
  // multiply its own idf).
  //
  // Compound identifiers are handled asymmetrically to the index on purpose.
  // Indexing emits `bound_relax_factor` AND its parts, so that a reader who
  // types the name as three words still finds it. Querying must not: expanded
  // into {bound_relax_factor, bound, relax, factor}, the query "bound_relax_
  // factor" was won by a passage about `fix_relax` that matched three of the
  // four parts and not the name. So an identifier the corpus actually
  // contains is searched as itself, and only an unknown one falls back to its
  // parts.
  function queryTerms(idx, query) {
    var raw = String(query).toLowerCase().match(/[a-z0-9_]+/g) || [];
    var seen = Object.create(null);
    var out = [];

    function push(tok) {
      if (tok && !seen[tok]) {
        seen[tok] = 1;
        out.push(tok);
      }
    }

    for (var i = 0; i < raw.length; i++) {
      var t = raw[i];
      if (STOPWORDS[t]) continue;
      var whole = normTok(t);
      if (t.indexOf("_") > 0 && !idx.df[whole]) {
        var parts = t.split("_");
        for (var j = 0; j < parts.length; j++) {
          if (parts[j] && !STOPWORDS[parts[j]]) push(normTok(parts[j]));
        }
      } else {
        push(whole);
      }
    }
    return out;
  }

  function search(idx, query, limit) {
    var qterms = queryTerms(idx, query);
    if (!qterms.length) return [];

    var scored = [];
    for (var i = 0; i < idx.N; i++) {
      var doc = idx.docs[i];
      var score = 0;
      for (var q = 0; q < qterms.length; q++) {
        var term = qterms[q];
        var f = doc.tf[term];
        if (!f) continue;
        var n = idx.df[term] || 0;
        var idf = Math.log(1 + (idx.N - n + 0.5) / (n + 0.5));
        var denom = f + K1 * (1 - B + (B * doc.len) / idx.avgdl);
        score += idf * ((f * (K1 + 1)) / denom);
        if (doc.head[term]) score += HEAD_BONUS * idf;
        if (doc.title[term]) score += TITLE_BONUS * idf;
      }
      // A coordination factor (scaling by the fraction of query terms a
      // passage matched) was tried here and removed: ablated, it moved
      // neither configuration of the eval set — 17/20 book-only and 18/20
      // with the wiki, either way. Once stopwords stopped inflating the term
      // count and the heading bonus stopped rewarding nesting depth, it had
      // nothing left to correct.
      if (score > 0) scored.push({ i: i, score: score });
    }

    scored.sort(function (a, b) {
      return b.score - a.score;
    });

    // Two levels of diversification, both of which showed up as visible
    // defects before they were added:
    //
    //   by anchor  a long section is split into several chunks, and they
    //              score alike, so the list showed the *same heading and the
    //              same link* twice in a row — which reads as a bug and
    //              spends two of the five context slots on one section.
    //   by page    three slices of one page is a worse context window, and a
    //              worse thing to read, than three pages that each mention
    //              the term.
    var seenUrl = Object.create(null);
    var perPage = Object.create(null);
    var hits = [];
    for (var s = 0; s < scored.length && hits.length < limit; s++) {
      var chunk = idx.chunks[scored[s].i];
      if (seenUrl[chunk.u]) continue;
      var page = chunk.u.split("#")[0];
      if ((perPage[page] || 0) >= 2) continue;
      seenUrl[chunk.u] = 1;
      perPage[page] = (perPage[page] || 0) + 1;
      hits.push({ chunk: chunk, score: scored[s].score });
    }
    return hits;
  }

  function loadIndex() {
    if (state.index) return Promise.resolve(state.index);
    if (state.loading) return state.loading;
    state.loading = fetch(indexUrl().href, { cache: "default" })
      .then(function (r) {
        if (!r.ok) throw new Error("index HTTP " + r.status);
        return r.json();
      })
      .then(function (doc) {
        state.index = buildIndex(doc.chunks || []);
        state.index.counts = doc.counts || null;
        return state.index;
      })
      .catch(function (err) {
        state.loading = null;
        throw err;
      });
    return state.loading;
  }

  // ------------------------------------------------------------------- ui --

  var ui = {};

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function injectStylesheet() {
    if (document.getElementById("discopt-ask-css")) return;
    var link = document.createElement("link");
    link.id = "discopt-ask-css";
    link.rel = "stylesheet";
    // Alongside this script, so it reaches archived books the same way.
    link.href = new URL("ask.css", self().src).href;
    document.head.appendChild(link);
  }

  function setStatus(text, kind) {
    if (!ui.status) return;
    ui.status.textContent = text || "";
    ui.status.className = "discopt-ask-status" + (kind ? " discopt-ask-status-" + kind : "");
  }

  function openPanel() {
    injectStylesheet();
    ui.panel.hidden = false;
    document.body.classList.add("discopt-ask-open");
    ui.input.focus();
    loadIndex()
      .then(function (idx) {
        if (!ui.askBtn.dataset.ready) {
          ui.askBtn.dataset.ready = "1";
          ui.askBtn.disabled = false;
          var c = idx.counts;
          setStatus(
            c
              ? "Ready — " + c.total + " passages from " + c.pages + " pages."
              : "Ready — " + idx.N + " passages."
          );
        }
      })
      .catch(function (err) {
        setStatus("Could not load the search index (" + err.message + ").", "error");
      });
  }

  function closePanel() {
    ui.panel.hidden = true;
    document.body.classList.remove("discopt-ask-open");
    ui.toggle.focus();
  }

  // A speech bubble, inline rather than a font glyph or an image: the button is
  // injected into archived books whose CSS has no icon font, and an <img>
  // would be a second request resolved against a path this file would have to
  // compute. 16x16 viewBox, currentColor fill, so it inherits the pill.
  var TOGGLE_ICON =
    '<svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg" ' +
    'aria-hidden="true" focusable="false">' +
    '<path d="M8 1.5c-3.9 0-7 2.4-7 5.4 0 1.7 1 3.2 2.6 4.2-.1.8-.5 1.8-1.3 2.6' +
    '-.2.2 0 .5.2.5 1.6-.1 3-.7 4-1.5.5.1 1 .1 1.5.1 3.9 0 7-2.4 7-5.4S11.9 1.5 8 1.5z"/>' +
    "</svg>";

  function buildToggle() {
    // The button is fixed to the viewport, so this is not a mount point — it
    // is the test for "this page is a rendered book page". The theme puts the
    // content in <article class="bd-article">; a 404, a print view or a bare
    // redirect stub has none, and there the panel is unreachable, so init()
    // drops it rather than leaving a dialog in the DOM.
    var chrome = document.querySelector("article.bd-article") || document.querySelector("main");
    if (!chrome || document.getElementById("discopt-ask-toggle")) return false;
    var btn = el("button", "discopt-ask-toggle");
    btn.id = "discopt-ask-toggle";
    btn.type = "button";
    btn.title = "Ask a question about the discopt docs (runs in your browser)";
    btn.setAttribute("aria-label", "Ask the docs assistant");
    var icon = el("span", "discopt-ask-toggle-icon");
    icon.setAttribute("aria-hidden", "true");
    icon.innerHTML = TOGGLE_ICON;
    btn.appendChild(icon);
    btn.appendChild(el("span", "discopt-ask-toggle-label", "Ask"));
    btn.addEventListener("click", function () {
      if (ui.panel.hidden) openPanel();
      else closePanel();
    });
    document.body.appendChild(btn);
    ui.toggle = btn;
    return true;
  }

  function modelOptions(useF16) {
    var quant = useF16 ? "q4f16_1" : "q4f32_1";
    return MODELS.map(function (m) {
      return {
        id: m.base + "-" + quant + "-MLC",
        // f32 weights are roughly a third larger than the f16 figure; say the
        // number for the build the reader will actually download.
        label: m.label + " (~" + Math.round((useF16 ? m.vram : m.vram * 1.3) / 100) / 10 + " GB)"
      };
    });
  }

  function buildPanel() {
    var panel = el("aside", "discopt-ask-panel");
    panel.id = "discopt-ask-panel";
    panel.hidden = true;
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-label", "discopt docs assistant");

    var head = el("div", "discopt-ask-head");
    head.appendChild(el("h2", "discopt-ask-title", "Ask discopt"));
    var close = el("button", "discopt-ask-close", "×");
    close.type = "button";
    close.setAttribute("aria-label", "Close");
    close.addEventListener("click", closePanel);
    head.appendChild(close);
    panel.appendChild(head);

    ui.status = el("div", "discopt-ask-status", "Loading the search index…");
    panel.appendChild(ui.status);

    // --- question -----------------------------------------------------
    var form = el("form", "discopt-ask-form");
    ui.input = el("textarea", "discopt-ask-input");
    ui.input.rows = 2;
    ui.input.placeholder = "e.g. why does relaxing the bounds cost iterations?";
    ui.input.setAttribute("aria-label", "Your question");
    // Enter submits, Shift+Enter newlines — the panel is a question box, not
    // an editor.
    ui.input.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        form.dispatchEvent(new Event("submit", { cancelable: true }));
      }
    });
    form.appendChild(ui.input);

    ui.askBtn = el("button", "discopt-ask-submit", "Ask");
    ui.askBtn.type = "submit";
    ui.askBtn.disabled = true;
    form.appendChild(ui.askBtn);

    form.addEventListener("submit", function (e) {
      e.preventDefault();
      ask(ui.input.value.trim());
    });
    panel.appendChild(form);

    // --- model --------------------------------------------------------
    ui.modelRow = el("div", "discopt-ask-model");
    panel.appendChild(ui.modelRow);

    ui.answer = el("div", "discopt-ask-answer");
    panel.appendChild(ui.answer);

    ui.sources = el("div", "discopt-ask-sources");
    panel.appendChild(ui.sources);

    var foot = el("div", "discopt-ask-foot");
    foot.appendChild(
      document.createTextNode(
        "Searches these docs. Answers, when a model is loaded, " +
          "are generated on your own device — nothing you type is sent anywhere. "
      )
    );
    var docLink = el("a", null, "How this works");
    docLink.href = new URL("ask.html", siteRoot()).href;
    foot.appendChild(docLink);
    foot.appendChild(document.createTextNode("."));
    panel.appendChild(foot);

    document.body.appendChild(panel);
    ui.panel = panel;
    buildModelRow();
  }

  function noGpuNote(why) {
    ui.modelRow.textContent = "";
    ui.modelRow.appendChild(
      el(
        "span",
        "discopt-ask-note",
        why + " — showing matching passages only. Written answers need WebGPU: " +
          "a recent Chrome or Edge, Safari 26+, or Firefox with WebGPU enabled."
      )
    );
  }

  function buildModelRow() {
    // `navigator.gpu` existing is NOT the same as WebGPU working:
    // requestAdapter() resolves *null* on a machine with no usable GPU
    // (headless browsers, blocklisted drivers, some VMs). Offering the button
    // anyway means the reader waits, then gets an error from deep inside
    // WebLLM. Ask the adapter first and say so up front instead.
    if (!navigator.gpu) {
      noGpuNote("No WebGPU in this browser");
      return;
    }
    ui.modelRow.textContent = "";
    ui.modelRow.appendChild(el("span", "discopt-ask-note", "Checking for a usable GPU…"));

    navigator.gpu
      .requestAdapter()
      .then(function (adapter) {
        if (!adapter) {
          noGpuNote("No usable GPU available to this browser");
          return;
        }
        var f16 = !!(adapter.features && adapter.features.has("shader-f16"));
        renderModelPicker(modelOptions(f16));
      })
      .catch(function (err) {
        noGpuNote("WebGPU could not be initialized (" + (err && err.message ? err.message : err) + ")");
      });
  }

  function renderModelPicker(opts) {
    var row = ui.modelRow;
    row.textContent = "";

    var sel = el("select", "discopt-ask-select");
    sel.setAttribute("aria-label", "Local model");

    var remembered = null;
    try {
      remembered = localStorage.getItem(LS_MODEL);
    } catch (e) {
      /* private mode: fall through to the default (the first, smallest) */
    }
    opts.forEach(function (o) {
      var opt = el("option", null, o.label);
      opt.value = o.id;
      if (o.id === remembered) opt.selected = true;
      sel.appendChild(opt);
    });

    var btn = el("button", "discopt-ask-load", "Load model");
    btn.type = "button";
    btn.addEventListener("click", function () {
      loadModel(sel.value, btn);
    });

    row.appendChild(sel);
    row.appendChild(btn);
    row.appendChild(
      el(
        "span",
        "discopt-ask-note",
        "Downloaded once, then cached by your browser. Until then you get passages, not prose."
      )
    );
    ui.modelSelect = sel;
  }

  async function loadModel(modelId, btn) {
    if (!modelId) return;
    btn.disabled = true;
    if (ui.modelSelect) ui.modelSelect.disabled = true;
    setStatus("Fetching the WebLLM runtime…", "busy");
    try {
      var webllm = await import(/* webpackIgnore: true */ WEBLLM_URL);
      state.engine = await webllm.CreateMLCEngine(modelId, {
        initProgressCallback: function (p) {
          setStatus(p && p.text ? p.text : "Loading model…", "busy");
        }
      });
      state.engineModel = modelId;
      try {
        localStorage.setItem(LS_MODEL, modelId);
      } catch (e) {
        /* private mode: the model still works, it just is not remembered */
      }
      setStatus("Model ready — answers will be written, with citations.", "ok");
      btn.textContent = "Model loaded";
      if (ui.modelSelect) ui.modelSelect.disabled = false;
    } catch (err) {
      // Leave retrieval fully working; this is an enhancement that failed.
      state.engine = null;
      btn.disabled = false;
      if (ui.modelSelect) ui.modelSelect.disabled = false;
      setStatus(
        "Could not load the model (" + (err && err.message ? err.message : err) + "). Passages still work.",
        "error"
      );
    }
  }

  // --------------------------------------------------------------- answer --

  function renderSources(hits) {
    ui.sources.textContent = "";
    if (!hits.length) return;
    ui.sources.appendChild(el("h3", "discopt-ask-sources-title", "Passages"));
    var ol = el("ol", "discopt-ask-source-list");
    hits.forEach(function (hit) {
      var li = el("li");
      var a = el("a", "discopt-ask-source-link", hit.chunk.h);
      a.href = hrefFor(hit.chunk);
      li.appendChild(a);
      li.appendChild(el("p", "discopt-ask-excerpt", excerpt(hit.chunk.x)));
      ol.appendChild(li);
    });
    ui.sources.appendChild(ol);
  }

  // The index stores markdown, because that is what the model reads best —
  // a fenced block and a bolded caveat are signal to it. A human skimming a
  // 260-character preview gets the opposite: `**Have an analytic Hessian?**`
  // and stray `*` bullets are noise. So the preview, and only the preview,
  // is flattened to prose.
  function excerpt(text, max) {
    var flat = text
      .replace(/^\s*(```+|~~~+).*$/gm, " ") // fence lines
      .replace(/^\s{0,3}#{1,6}\s+/gm, "") // stray headings
      .replace(/^\s{0,3}[-*+]\s+/gm, "· ") // bullets
      .replace(/^\s{0,3}>\s?/gm, "") // block quotes
      .replace(/\*\*([^*]+)\*\*/g, "$1")
      .replace(/(^|\s)\*([^*\n]+)\*(?=\s|$|[.,;:)])/g, "$1$2")
      .replace(/`+/g, "")
      .replace(/!?\[([^\]]*)\]\([^)]*\)/g, "$1") // links and images
      .replace(/\s+/g, " ")
      .trim();
    max = max || 260;
    return flat.length > max ? flat.slice(0, max) + "…" : flat;
  }

  function buildPrompt(question, hits) {
    var context = hits
      .map(function (hit, n) {
        var body = hit.chunk.x;
        if (body.length > MAX_CTX_CHARS) body = body.slice(0, MAX_CTX_CHARS) + "…";
        return "[" + (n + 1) + "] " + hit.chunk.h + "\n" + body;
      })
      .join("\n\n");

    return [
      {
        role: "system",
        content:
          "You are the discopt documentation assistant. discopt is a hybrid " +
          "Mixed-Integer Nonlinear Programming (MINLP) solver with a Rust core: " +
          "spatial branch-and-bound, McCormick and alphaBB relaxations, an " +
          "in-house simplex, and a Python modeling API.\n\n" +
          "Answer ONLY from the numbered excerpts the user provides. If they do " +
          "not contain the answer, say so plainly and name the excerpt that looks " +
          "closest — do not fill the gap from memory. Quote option names, values " +
          "and status strings exactly as written. Cite every claim with the " +
          "bracketed number of the excerpt it came from, like [2]. Be concise."
      },
      {
        role: "user",
        content: "Excerpts:\n\n" + context + "\n\nQuestion: " + question
      }
    ];
  }

  // Model output is rendered as text, never as HTML: inline `code` becomes a
  // <code> element and everything else stays a text node, so a stray angle
  // bracket in an answer cannot become markup.
  function renderAnswer(text) {
    ui.answer.textContent = "";
    var para = el("p", "discopt-ask-answer-body");
    var parts = text.split(/(`[^`]+`)/);
    parts.forEach(function (part) {
      if (part.length > 2 && part.charAt(0) === "`" && part.charAt(part.length - 1) === "`") {
        para.appendChild(el("code", null, part.slice(1, -1)));
      } else if (part) {
        para.appendChild(document.createTextNode(part));
      }
    });
    ui.answer.appendChild(para);
  }

  async function ask(question) {
    if (!question || state.busy) return;
    state.busy = true;
    ui.askBtn.disabled = true;
    ui.answer.textContent = "";
    ui.sources.textContent = "";

    try {
      var idx = await loadIndex();
      var hits = search(idx, question, TOP_SHOW);
      state.lastHits = hits;

      if (!hits.length) {
        setStatus("");
        renderAnswer("Nothing in the docs matched that. Try an API name (dm.Model, add_predictor) or the exact status string.");
        return;
      }

      renderSources(hits);

      if (!state.engine) {
        setStatus(
          navigator.gpu
            ? "Showing passages. Load a model above for a written answer."
            : "Showing passages (no WebGPU in this browser)."
        );
        return;
      }

      setStatus("Thinking…", "busy");
      var messages = buildPrompt(question, hits.slice(0, TOP_K));
      var stream = await state.engine.chat.completions.create({
        messages: messages,
        stream: true,
        temperature: 0.2,
        max_tokens: 600
      });

      var acc = "";
      for await (var part of stream) {
        var delta = part && part.choices && part.choices[0] && part.choices[0].delta;
        if (delta && delta.content) {
          acc += delta.content;
          renderAnswer(acc);
        }
      }
      setStatus("Answered from the passages below.", "ok");
    } catch (err) {
      setStatus("Failed: " + (err && err.message ? err.message : err), "error");
    } finally {
      state.busy = false;
      ui.askBtn.disabled = false;
    }
  }

  // ----------------------------------------------------------------- init --

  function init() {
    injectStylesheet();
    buildPanel();
    if (!buildToggle()) {
      // Not a rendered book page (404, print view, redirect stub): drop the
      // panel too rather than leave an unreachable dialog in the DOM.
      if (ui.panel && ui.panel.parentNode) ui.panel.parentNode.removeChild(ui.panel);
      return;
    }
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && ui.panel && !ui.panel.hidden) closePanel();
    });
  }

  // Test seam. The retrieval half — tokenizer, stemmer, index, scorer — is
  // pure, and docs/tests/ask_retrieval.mjs exercises it under node against a
  // freshly built index. Requiring THIS file rather than a copy of the
  // scoring is the point: a test of a reimplementation would go green while
  // the shipped ranker regressed. There is no `document` in node, so the DOM
  // half is simply never reached.
  if (typeof document === "undefined") {
    if (typeof module !== "undefined" && module.exports) {
      module.exports = {
        STOPWORDS: STOPWORDS,
        normTok: normTok,
        tokenize: tokenize,
        buildIndex: buildIndex,
        queryTerms: queryTerms,
        search: search,
        buildPrompt: buildPrompt,
        MAX_CTX_CHARS: MAX_CTX_CHARS
      };
    }
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
