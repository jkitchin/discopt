// Guard for the docs assistant's retrieval half (docs/_static/ask.js).
//
// It loads the SHIPPED ask.js — not a copy of the scoring — through the test
// seam at the bottom of that file, and runs it against an index built from the
// live book. A test of a reimplemented ranker would stay green while the
// ranker readers actually use regressed, which is the whole failure mode here:
// retrieval has no exception to throw. It returns the wrong thing, silently,
// and the only symptom is a reader not finding the page.
//
// Two things are checked, and they fail for different reasons:
//
//   1. The stemmer, against known Porter step-1 outputs and against the
//      specific pairs this corpus needs collapsed. This is a unit test: it
//      does not depend on what the docs say.
//   2. Ranking, against a labelled query set. This is a *corpus* test and it
//      is deliberately loose — see THRESHOLDS below.
//
// Usage:
//   make docs                                   # the index is built from HTML
//   python3 scripts/build-docs-index.py -o /tmp/ask-index.json --quiet
//   node docs/tests/ask_retrieval.mjs /tmp/ask-index.json
//
// `make ask-check` does both steps.

import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const ask = require(resolve(here, "../_static/ask.js"));

const indexPath = process.argv[2];
if (!indexPath) {
  console.error("usage: node docs/tests/ask_retrieval.mjs <ask-index.json>");
  process.exit(2);
}

let failures = 0;
let checks = 0;
function check(ok, msg) {
  checks++;
  if (!ok) failures++;
  console.log((ok ? "  ok   " : "  FAIL ") + msg);
}

// ---------------------------------------------------------------- stemmer --

console.log("stemmer: pairs this corpus needs collapsed");
for (const [a, b] of [
  ["relaxing", "relax"],
  ["relaxed", "relax"],
  ["branching", "branch"],
  ["branched", "branch"],
  // The +e restoration case: without it "scaling" stems to `scal` and stops
  // matching a page called "Scaling".
  ["scaling", "scale"],
  ["iterations", "iteration"],
  ["solves", "solve"],
  ["tightening", "tighten"],
  ["bounded", "bound"],
  ["duals", "dual"],
  ["constraints", "constraint"],
  ["variables", "variable"]
]) {
  check(ask.normTok(a) === ask.normTok(b), a + " ≡ " + b + " → " + ask.normTok(a));
}

console.log("stemmer: canonical Porter step-1 outputs");
for (const [w, want] of [
  ["caresses", "caress"],
  ["ponies", "poni"],
  ["cats", "cat"],
  ["feed", "feed"],
  ["agreed", "agree"],
  ["plastered", "plaster"],
  ["motoring", "motor"],
  ["hopping", "hop"],
  ["failing", "fail"],
  ["filing", "file"],
  ["happy", "happi"],
  ["sky", "sky"],
  // Known and accepted: Porter step 1 does not restore the `e` here, because
  // the cvc rule only fires at m == 1 and `converg` has m == 2. Pinned so the
  // asymmetry is a decision on the record rather than a surprise.
  ["converged", "converg"],
  ["converge", "converge"]
]) {
  check(ask.normTok(w) === want, w + " → " + ask.normTok(w));
}

console.log("stemmer: identifiers and short tokens pass through untouched");
for (const w of [
  "add_predictor",
  "subject_to",
  "register_function",
  "minlp",
  "gdp",
  "lp",
  "qp",
  "dae"
]) {
  check(ask.normTok(w) === w, w + " unchanged");
}

console.log("distinct terms must not collide");
for (const [a, b] of [
  ["scaling", "scalar"],
  ["convex", "converge"],
  ["dual", "duel"],
  ["bound", "bind"],
  ["relax", "relay"]
]) {
  check(ask.normTok(a) !== ask.normTok(b), a + " ≠ " + b);
}

// ---------------------------------------------------------------- ranking --

const doc = JSON.parse(readFileSync(indexPath, "utf8"));
const idx = ask.buildIndex(doc.chunks || []);
console.log("\nindex: " + idx.N + " passages, avg " + idx.avgdl.toFixed(0) + " tokens");
check(idx.N > 500, "index is populated");

// Every citation must be able to deep-link. The index builder enforces this
// too, but it is cheap to re-assert against the artifact that actually ships:
// a chunk with no anchor drops its reader at the top of a long notebook.
{
  const anchored = (doc.chunks || []).filter((c) => c.u.includes("#")).length;
  const pct = (100 * anchored) / (doc.chunks || []).length;
  check(pct > 95, "citations are anchored (" + pct.toFixed(1) + "%)");
}

// Notebooks are two thirds of this book. An extractor change that silently
// stopped reading them would leave a plausible-looking index that cannot
// answer about any tutorial.
{
  const nb = (doc.chunks || []).filter((c) => c.u.startsWith("notebooks/")).length;
  check(nb > 200, "notebook pages are indexed (" + nb + " passages)");
}

// Code must survive as code. Pygments wraps every token in its own <span>, and
// an extractor that joins those with whitespace turns `dm.Model` into three
// tokens — which still "works" and still returns hits, just never for the name
// the reader typed.
{
  const joined = (doc.chunks || []).map((c) => c.x).join(" ");
  check(joined.includes("dm.Model"), "dotted API names survive extraction");
  check(joined.includes("subject_to"), "underscored API names survive extraction");
}

console.log("\nquery-side compound handling");
{
  // An identifier the corpus contains is searched as itself, never split —
  // splitting lets a passage that happens to contain `predictor` outrank the
  // documentation of `add_predictor`.
  const known = ask.queryTerms(idx, "add_predictor");
  check(
    known.length === 1 && known[0] === "add_predictor",
    "known identifier stays whole: [" + known.join(", ") + "]"
  );
  const unknown = ask.queryTerms(idx, "no_such_option_here");
  check(unknown.length > 1, "unknown identifier falls back to parts: [" + unknown.join(", ") + "]");
  const stopped = ask.queryTerms(idx, "what does it do");
  check(stopped.length === 0, "an all-stopword question yields no terms");
}

console.log("\nresult shaping");
{
  const hits = ask.search(idx, "relaxation", 6);
  const urls = hits.map((h) => h.chunk.u);
  check(new Set(urls).size === urls.length, "no two hits share a URL");
  const pages = urls.map((u) => u.split("#")[0]);
  const worst = Math.max(...pages.map((p) => pages.filter((q) => q === p).length));
  check(worst <= 2, "at most two hits per page (worst: " + worst + ")");
  check(ask.search(idx, "", 6).length === 0, "empty query returns nothing");
  check(ask.search(idx, "zzzqqqxxnotaword", 6).length === 0, "unmatched query returns nothing");
}

// Labelled set. `want` substrings are matched against the citation URL, and a
// query lists every page that genuinely answers it — not one blessed page —
// because several of these questions have more than one honest home in a book
// that is mostly tutorials.
//
// THRESHOLDS ARE DELIBERATELY BELOW THE MEASURED SCORE. This is a corpus
// test: adding or retitling a page can legitimately move a ranking, and a
// guard pinned to the exact current number would fail on honest doc edits and
// get raised until it meant nothing. It is set to catch the shape of a real
// regression — a scoring change that drops several queries at once — not
// single-rank drift.
//
// Measured on the v0.9.0 book (1110 passages from 86 pages): 18/20 top-1,
// 20/20 top-5. The floors below sit well under that. The two queries that
// miss top-1 both return an honest neighbour — "how do I add an integer
// variable" lands on the MIQP tutorial's integer section, and "fitting model
// parameters" on the sIPOPT curve-fitting example — so they are recorded here
// rather than papered over by widening `want`.
const EVAL = [
  { q: "how do I warm start a solve", want: ["warm_start", "tutorial_solver_selection"] },
  { q: "what is feasibility based bound tightening", want: ["bound_tightening", "presolve"] },
  { q: "embedding a trained neural network in a model", want: ["nn_embedding", "neural_dae"] },
  {
    q: "generalized disjunctive programming with big-M",
    want: ["tutorial_gdp", "disjunction_semantics"]
  },
  { q: "outer approximation algorithm for convex MINLP", want: ["tutorial_oa", "mip_nlp"] },
  { q: "Benders decomposition", want: ["tutorial_benders", "tutorial_gbd", "decomposition_advisor"] },
  {
    q: "McCormick relaxation envelopes for bilinear terms",
    want: ["symbolic_envelopes", "global_optimization", "tutorial_minlp", "solver_internals"]
  },
  {
    q: "how do I add an integer variable",
    want: ["quickstart", "modeling_guide", "tutorial_milp", "problem_classes"]
  },
  { q: "using discopt from pyomo", want: ["pyomo_solver"] },
  { q: "exporting a model to nl format", want: ["export_formats", "model_serialization"] },
  {
    q: "solving a DAE with orthogonal collocation",
    want: ["tutorial_dae", "neural_dae", "cstr_fit_doe_optimize"]
  },
  { q: "multiobjective pareto front", want: ["tutorial_multiobjective"] },
  {
    q: "fitting model parameters to experimental data",
    want: ["tutorial_estimation", "estimation", "cstr_fit_doe_optimize"]
  },
  { q: "why is my model infeasible", want: ["infeasibility_iis", "conflict_analysis"] },
  { q: "detecting whether a problem is convex", want: ["convexity_detection", "convex_fast_path"] },
  { q: "robust optimization with an uncertainty set", want: ["tutorial_robust", "robust_adjustable"] },
  { q: "explain a solve result with an LLM", want: ["llm_features", "llm_integration"] },
  { q: "sensitivity analysis of the solution", want: ["sensitivity_analysis", "tutorial_pounce_sipopt"] },
  { q: "primal heuristics for finding an incumbent", want: ["primal_heuristics", "solver_internals"] },
  { q: "bilevel optimization with KKT conditions", want: ["bilevel", "complementarity"] }
];

const THRESHOLD_TOP1 = 13;
const THRESHOLD_TOP5 = 17;

console.log("\nprompt construction (the RAG contract)");
{
  const hits = ask.search(idx, "McCormick relaxation", 5);
  const msgs = ask.buildPrompt("what is a McCormick relaxation?", hits);
  check(
    msgs.length === 2 && msgs[0].role === "system" && msgs[1].role === "user",
    "system + user message"
  );
  // The grounding instructions are the only thing keeping a 1B model from
  // answering about some other solver from memory; losing them would not fail
  // loudly.
  check(/ONLY from the numbered excerpts/.test(msgs[0].content), "system prompt grounds the model");
  check(/Cite every claim/.test(msgs[0].content), "system prompt demands citations");
  check(
    msgs[1].content.includes("what is a McCormick relaxation?"),
    "question is in the user message"
  );
  for (let n = 1; n <= hits.length; n++) {
    check(msgs[1].content.includes("[" + n + "] "), "excerpt [" + n + "] is numbered");
  }
  // Every passage must be reachable from the source list the reader sees, or
  // a citation points at nothing.
  check(hits.length <= 6, "no more passages than the panel lists");

  // Truncation: a long passage must be cut, not sent whole, or five of them
  // overflow a small model's context.
  const long = [
    { chunk: { h: "H", t: "T", u: "u.html", k: "book", x: "x".repeat(ask.MAX_CTX_CHARS * 3) } }
  ];
  const cut = ask.buildPrompt("q", long)[1].content;
  check(cut.length < ask.MAX_CTX_CHARS * 2, "over-long passage is truncated (" + cut.length + " chars)");
  check(cut.includes("…"), "truncation is marked with an ellipsis");
}

console.log("\nranking over " + EVAL.length + " labelled queries");
let top1 = 0;
let top5 = 0;
for (const c of EVAL) {
  const urls = ask.search(idx, c.q, 6).map((h) => h.chunk.u);
  const hit = (u) => c.want.some((w) => u.includes(w));
  const at1 = urls.length > 0 && hit(urls[0]);
  const at5 = urls.slice(0, 5).some(hit);
  if (at1) top1++;
  if (at5) top5++;
  console.log("  " + (at1 ? "T1" : at5 ? ".5" : "XX") + "  " + c.q);
  if (!at1) console.log("        got: " + (urls[0] || "(nothing)"));
}

console.log("\n  top-1 " + top1 + "/" + EVAL.length + " (floor " + THRESHOLD_TOP1 + ")");
console.log("  top-5 " + top5 + "/" + EVAL.length + " (floor " + THRESHOLD_TOP5 + ")");
check(top1 >= THRESHOLD_TOP1, "top-1 above the floor");
check(top5 >= THRESHOLD_TOP5, "top-5 above the floor");

// ---- The answer reader ---------------------------------------------------
//
// `parseAnswer` is the pure half of renderAnswer(): text in, block structure
// out, no DOM. It is tested here rather than in the browser suite because the
// browser suite cannot get a model to speak -- a 900 MB WebGPU download is not
// a CI step -- so without this the markdown and TeX handling has no guard at
// all and regresses to the flat `\[ … \]` a reader reported.
console.log("\nanswer rendering");

const ANSWER = [
  "## Quadratic programming",
  "",
  "QP is an optimization problem of the form",
  "",
  "\\[",
  "\\min_{x} \\quad \\tfrac{1}{2} x^\\top Q x + c^\\top x",
  "\\]",
  "",
  "where \\(Q\\) is symmetric [1]. Use `dm.Model` to build one [2, 3].",
  "",
  "- **Convex** when $Q \\succeq 0$ [1]",
  "- Nonconvex otherwise",
  "",
  "```python",
  "m = dm.Model()",
  "```"
].join("\n");

const blocks = ask.parseAnswer(ANSWER);
const types = blocks.map((b) => b.type);
check(types.includes("heading"), "a `## ` line becomes a heading, not prose");
check(types.includes("math"), "a display \\[ … \\] block is lifted out of the prose");
check(types.includes("list"), "a `- ` run becomes a list");
check(types.includes("code"), "a fenced block becomes code");

const math = blocks.find((b) => b.type === "math");
check(
  math && math.text.includes("\\min_{x}") && !math.text.includes("\\["),
  "the math block carries bare TeX, with its delimiters stripped"
);

const code = blocks.find((b) => b.type === "code");
check(code && code.lang === "python", "the fence language is kept");
check(code && code.text === "m = dm.Model()", "the fenced body survives verbatim");

// Citations are the point of the exercise: they have to come out as numbers
// the renderer can turn into links, including a grouped `[2, 3]`.
const flat = blocks
  .filter((b) => b.inline)
  .reduce((a, b) => a.concat(b.inline), [])
  .concat(
    blocks
      .filter((b) => b.items)
      .reduce((a, b) => a.concat(b.items.reduce((x, y) => x.concat(y), [])), [])
  );
const cites = flat.filter((t) => t.t === "cite").map((t) => t.n);
check(cites.length >= 4, "citations are tokenized, not left as text (" + cites.length + ")");
check(
  cites.includes(1) && cites.includes(2) && cites.includes(3),
  "a grouped [2, 3] becomes two citations: " + JSON.stringify(cites)
);
check(
  flat.some((t) => t.t === "math" && t.v.includes("Q")),
  "inline \\( … \\) and $ … $ become math tokens"
);
check(
  flat.some((t) => t.t === "code" && t.v === "dm.Model"),
  "inline `code` survives"
);
check(
  flat.some((t) => t.t === "strong" && t.v === "Convex"),
  "**bold** survives inside a list item"
);

// Prices are not equations. Without the guard in parseInline this eats the
// prose between two dollar amounts and renders it as TeX.
const money = ask.parseInline("it costs $12, or $30 with support");
check(
  !money.some((t) => t.t === "math"),
  "a dollar amount is not mistaken for inline math"
);

// Half-written TeX and unterminated fences are the NORMAL mid-stream state,
// since renderAnswer runs on every delta. They must not throw.
let streamed = 0;
for (let i = 1; i <= ANSWER.length; i += 7) {
  ask.parseAnswer(ANSWER.slice(0, i));
  streamed++;
}
check(streamed > 20, "every prefix of a streaming answer parses (" + streamed + " prefixes)");

// The passage previews under an answer cannot be typeset -- they are cut at
// 260 characters, often mid-expression -- so they drop display math and unwrap
// inline math instead of showing the reader raw TeX.
const QP = ask.excerpt(
  "A quadratic program (QP) is an optimization problem of the form \\[ " +
    "\\min_{x} \\quad \\tfrac{1}{2} x^\\top Q\\, x + c^\\top x \\] " +
    "where \\(x \\in \\mathbb{R}^n\\) is the vector of decision variables."
);
check(!QP.includes("\\["), "a preview drops display-math delimiters: " + QP.slice(0, 60));
check(!QP.includes("\\tfrac"), "a preview drops TeX control sequences");
check(QP.includes("quadratic program"), "a preview keeps the prose around the math");

// The relations carry the sentence, so a short glyph map survives the strip.
// Without it "x \\in \\mathbb{R}^n" previews as "x R^n", which reads as a typo.
const REL = ask.excerpt(
  "The feasible set is every point with \\(x \\in \\mathbb{R}^n\\) and " +
    "\\(Q \\succeq 0\\) and \\(A x \\le b\\) holding simultaneously here."
);
check(REL.includes("\u2208"), "\\in survives as a glyph: " + REL.slice(0, 70));
check(REL.includes("\u2264"), "\\le survives as a glyph");
check(!/\\[a-zA-Z]/.test(REL), "no backslash command is left in a preview: " + REL);

// A passage that is almost entirely an equation must not preview as blank.
const MOSTLY = ask.excerpt("\\[ x^2 + y^2 = z^2 \\]");
check(MOSTLY.trim().length > 0, "an equation-only passage still previews something");

// texToProse: the fallback for an expression MathJax REJECTED. Distinct from
// excerpt/stripTex, which drops display math because prose carries the preview;
// here the expression IS the content, so every wrapper is unwrapped instead.
const BADENV = ask.texToProse("\\begin{split}\n\\min_x c^\\top x \\\\ Ax = b\n\\end{aligned}");
check(!BADENV.includes("\\"), "a mismatched environment leaves no backslash: " + BADENV);
check(!/begin|end\{/.test(BADENV), "the environment delimiters are gone: " + BADENV);
check(BADENV.includes("min"), "the operator name survives demotion: " + BADENV);
check(BADENV.includes("Ax = b"), "the body survives demotion: " + BADENV);

// stripTex would hand back the untouched source here (its <40-char guard), which
// is the raw TeX the reader was never meant to see. That is why this is separate.
check(
  ask.stripTex("\\begin{split} x \\end{aligned}").includes("\\begin"),
  "stripTex still returns short input untouched (the guard texToProse must not share)"
);
check(
  !ask.texToProse("\\begin{split} x \\end{aligned}").includes("\\begin"),
  "texToProse does not inherit that guard"
);

// A two-argument macro must not be flattened: \tfrac{1}{2} -> "12" would be a
// WRONG number on the page, the one outcome worse than unreadable text.
check(ask.texToProse("\\tfrac{1}{2} x").startsWith("1/2"), "\\tfrac renders as a ratio");
check(
  ask.texToProse("\\frac{UB - LB}{|UB|}").startsWith("(UB - LB)/"),
  "a compound numerator is parenthesised: " + ask.texToProse("\\frac{UB - LB}{|UB|}")
);
check(ask.texToProse("\\quad x \\quad").trim() === "x", "spacing macros become space");
check(ask.texToProse("\\left( x \\right)").includes("( x )"), "\\left/\\right are dropped");
check(ask.texToProse("") === "", "empty input returns empty, not undefined");
check(ask.texToProse("\\texttt{rel_gap}") === "rel_gap", "\\texttt is unwrapped");

// Prove the probe fired: a refactor that made every `check` unreachable would
// otherwise print a clean run and exit 0.
if (checks === 0) {
  console.log("\nask_retrieval: NO CHECKS EXECUTED");
  process.exit(2);
}

console.log(
  "\n" + (failures ? failures + " FAILURE(S) of " + checks : "ask_retrieval: OK (" + checks + " checks)")
);
process.exit(failures ? 1 : 0);
