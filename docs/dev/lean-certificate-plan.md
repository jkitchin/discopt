# Lean-checkable certificate of global optimality

Status: **Phase 0 (in progress)** — design + first vertical slice (Tier-1 feasibility).
Owner: correctness track. Tracking discussion: see this document's roadmap table.

## 1. Context and motivation

discopt's product is its **certificate**. When it reports `status="optimal"` with
`gap_certified=True` it is asserting two things: the incumbent is feasible (an honest
upper bound on a minimization objective), and no feasible point beats the dual bound
(a valid lower bound). Today that assertion lives only as in-memory numeric state — a
boolean `gap_certified` plus a finite `bound` on `SolveResult`
(`python/discopt/modeling/core.py`) — and the *witness* data that would justify it
(per-node relaxation coefficients, LP duals, Farkas rays, the leaf-box covering) is
computed per node and discarded. There is no artifact a third party can independently
check, and there was no formal-methods surface anywhere in the repo.

The goal is to make the certificate a **first-class, externally checkable object**:
discopt emits a machine-readable certificate, and a **Lean 4** development *proves*
(in the kernel) that the certificate implies the claim. This turns "trust our solver's
`gap_certified` flag" into "trust Lean's kernel plus our published, once-proven
checker." It serves Development Philosophy #1 (correctness is the product) and gives
the correctness backlog (`docs/dev/correctness-issues.md`, whose every entry is a
false-certificate class) a formal backstop: each known unsoundness (C-17, C-23, C-32,
C-34, …) becomes a proof obligation the checker cannot pass without discharging.

## 2. Core architectural decision: a verified *checker*, not per-instance proof terms

Two ways to get a Lean-backed guarantee:

- **Per-instance proof terms** — emit a bespoke Lean proof for each solved instance and
  have Lean elaborate it. Rejected: elaboration cost scales with tree size (thousands
  of nodes), and every instance re-pays proof search.
- **Verified checker (chosen)** — write **one** Lean function
  `checkCertificate : Model → Certificate → Bool` and prove **once** a soundness
  theorem: if it returns `true`, the claim holds. Per instance, Lean only *evaluates*
  the checker on the certificate (kernel reduction / `native_decide`), not searches for
  a proof. This is how certified SAT (LRAT/VeriPB), certified LP, and certified MILP
  checkers scale, and it matches the soundness-harness philosophy already in
  `discopt_benchmarks/utils/soundness.py` (validity = "a bound that exceeds the true
  optimum is a false certificate").

## 3. The top-level Lean soundness theorem (the target)

```
-- Model: variables with box bounds + integrality, constraints (body ⋈ rhs), objective, sense.
-- Feasible m x  ⇔  x ∈ box ∧ integral where required ∧ every constraint holds.
theorem checkCertificate_sound (m : Model) (c : Certificate) :
    checkCertificate m c = true →
      Feasible m c.incumbent
    ∧ objectiveValueMatches m c                          -- primal value is real
    ∧ (∀ x, Feasible m x → c.dualBound ≤ objective m x)   -- dual bound is a true global LB (min)
    ∧ c.incumbentValue ≤ c.dualBound + c.gapTol           -- gap closed ⇒ ε-global-optimal
```

Three **tiers** are three progressively stronger ways the checker establishes the
`∀ x, Feasible m x → dualBound ≤ objective m x` conjunct:

- **Tier 1 (feasibility / primal):** first two conjuncts only. Proves the incumbent is
  genuinely feasible with the stated value (an honest upper bound). No global claim.
- **Tier 2 (convex / KKT):** for models Lean can certify convex, a KKT point with valid
  multipliers is global — `dualBound = incumbentValue` justified by convexity.
- **Tier 3 (spatial B&B):** the general nonconvex proof — a leaf-box covering of the
  root box where every leaf supplies a valid lower bound ≥ `dualBound`, so the minimum
  over a cover of the domain is a global lower bound.

## 4. Certificate schema (what discopt emits)

JSON, `schema_version` bumped in `python/discopt/result_io.py`; the proof payload lives
under a top-level `certificate` key so existing solver-result consumers are unaffected.
Everything Lean must trust is emitted as **exact rationals** (`[num, den]` integer
pairs), never float repr — see §7. Sections:

1. **`model`** — a self-contained restatement mirroring the Rust `ModelRepr`
   (`crates/discopt-core/src/expr.rs`): a flat list of scalar **columns** (one per
   entry of each variable block, in `model._variables` order), each with `name`,
   `type ∈ {continuous, integer, binary}`, `lb`, `ub` (rationals or `null` for ±∞);
   `constraints` (each `{sense ∈ {le, eq, ge}, body: <expr>, rhs}`); `objective`
   (`{sense ∈ {min, max}, body: <expr>}`). The expression DAG is emitted node-by-node
   (§below).
2. **`incumbent`** — the flat point `x` (rationals, column order) + `objectiveValue`.
3. **`dualBound`** / **`tier`** / **`gapTol`** — the global lower bound (min sense), the
   tier claimed, and the gap tolerance. Tier-1 certificates omit `dualBound`.
4. **`tree`** (Tier 3) — the B&B leaves: each leaf's box, fathom reason, and witness
   (relaxation coefficients + LP dual for bound-fathomed; Farkas ray for infeasible;
   branch variable/split for internal nodes so the checker verifies **covering**).
5. **`relaxationWitness`** (Tier 3) — per-leaf affine under/over-estimators (McCormick,
   secant/tangent, α-BB α). These are closed-form in the leaf box bounds, so Lean can
   **recompute** them and check equality rather than trust emitted numbers.
6. **`convexWitness`** (Tier 2) — curvature tags + KKT multipliers.
7. **`meta`** — solver version, schema version, tolerances, source-`.nl` hash.

### Expression node encoding

Each `<expr>` is a tagged object:

| node | JSON |
|------|------|
| constant | `{"k":"const","v":[num,den]}` |
| variable (column `i`) | `{"k":"var","i":i}` |
| add/sub/mul/div/pow | `{"k":"add"|"sub"|"mul"|"div"|"pow","l":<expr>,"r":<expr>}` |
| neg / abs | `{"k":"neg"|"abs","x":<expr>}` |
| named function | `{"k":"fn","name":"exp"|…,"args":[<expr>,…]}` |

`pow` carries an integer exponent for the Tier-1 rational checker; `fn` nodes are
transcendental (`MathFunc`) and are **not** evaluable in exact ℚ — the Tier-1 checker
conservatively refuses any certificate whose checked expressions contain them (Phase 1
adds their interval enclosures over Mathlib reals).

## 5. Lean-side proof obligations (by tier)

A `lean/` Lake project. Modules:

- **`Discopt/Model.lean`** — `Expr`/`Constraint`/`Model` types + `eval`/`Feasible`
  semantics. Tier-1 evaluates over `ℚ` (exact, no Mathlib). Phase-1+ lifts the
  transcendental `MathFunc` to Mathlib's `Real.exp/log/sqrt/…`.
- **`Discopt/Checker.lean`** — `checkFeasible` (a `Bool` decision procedure) + the
  `checkFeasible_sound` theorem (Tier 1). Grows into `checkCertificate` across tiers.
- **`IntervalArith.lean`** (Phase 1) — verified enclosures `⟦f⟧(box) ⊇ {f(x):x∈box}`
  for every operator/`MathFunc`; discharges Tier-1 over the *full factorable* set and is
  reused everywhere.
- **`Envelopes.lean`** (Tier 3) — one validity lemma per relaxation family: McCormick
  (4 inequalities for `x·y` over a box), univariate `MathFunc` (convex ≥ tangent, ≤
  secant; concave mirror) tracking the `(cv,cc)` contract in
  `docs/design/relaxation-catalog.md §1–3, §5`, and α-BB. Each known-bad case in
  `correctness-issues.md` (e.g. C-32 inverted asin/acos curvature) is a lemma the
  envelope must satisfy or the checker rejects.
- **`LPDuality.lean`** (Tier 3) — LP weak duality: a dual-feasible `y ≥ 0` gives a valid
  leaf lower bound; a Farkas ray certifies emptiness. Exact over `ℚ`.
- **`Covering.lean`** (Tier 3) — branch splits are exhaustive (`x≤s ∨ x≥s`; integer
  `x≤⌊v⌋ ∨ x≥⌈v⌉`), so `min` of leaf bounds is a valid global lower bound.
- **`Convex.lean`** (Tier 2) — convex model + KKT ⇒ global.

## 6. Wiring on the discopt side

- **Emitter (Phase 0, done):** `python/discopt/certificate/` — `emit.py`
  (`build_feasibility_certificate(model, result)` walks the modeling-API DAG and the
  `SolveResult` incumbent; refuses loudly on unsupported nodes), `schema.py` (rationals
  + JSON), `refcheck.py` (a Python reference checker mirroring the Lean algorithm,
  exact over `fractions.Fraction`, used as a test oracle and to de-risk the Lean port).
  It reads only `model` + `SolveResult` — **no solver internals**, so it is
  bound-neutral by construction (the recorder below is a no-op path for Tier 1).
- **Retain per-node witnesses (Tier 3, later):** an opt-in recorder (default OFF, zero
  cost when off) capturing per fathomed leaf its box, fathom reason, relaxation witness,
  LP dual / Farkas ray, and branch decisions. Hook points: `TreeManager::process_evaluated`
  (`crates/discopt-core/src/bnb/tree_manager.rs`) and the `MccormickLPResult` return
  path (`python/discopt/_jax/mccormick_lp.py`, which already computes the
  Neumaier–Shcherbina `safe_bound`).
- **CLI (done):** `discopt solve --emit-certificate` writes `<stub>.cert.json`
  (re-loading the model via `from_nl`, as `--sol` does), and `discopt cert-check
  <file>` runs the reference checker (exit 0 ACCEPT / 1 REJECT). The Lean checker is
  the separate `lake exe check cert.json` step. (`result_io.py` is intentionally
  left untouched — it holds only a `SolveResult`, not the model the emitter needs.)

## 7. Risks and open questions

- **Float vs exact rational** — the crux. Lean proofs want exact arithmetic; solver math
  is float. Mitigation: emit *inputs* (box bounds, incumbent) as the exact rationals the
  floats denote (`Fraction(float)` is exact — a float is a dyadic rational) and recompute
  derived coefficients in Lean; formalize the Neumaier–Shcherbina directed rounding for
  any residual float slack in `LPDuality.lean`. Feasibility is checked with an exact
  rational tolerance.
- **Coverage discopt already refuses** — the `.nl` parser rejects floor/ceil/round/trunc/
  intdiv (C-5) and binary-format `.nl`. The certificate inherits these limits; the
  checker refuses the same class rather than silently pass.
- **Tree size** (Tier 3) — large certificates / kernel-eval time; mitigate with
  `native_decide` and compressed leaf encoding.
- **`gap_certified=False` paths** — NLP-BB heuristic mode (`nlp_bb`) is not a rigorous
  global proof; the emitter refuses to emit a Tier-2/3 certificate then, only a Tier-1
  feasibility certificate.
- **Faithfulness of the re-encoding** — Lean's `Model` must mean the same as discopt's;
  guarded by a differential test (Lean/`refcheck` eval vs discopt's evaluator).

## 8. Roadmap

| Phase | Deliverable | Proof reach |
|------|-------------|-------------|
| **0 (this effort)** | Design doc + Tier-1 emitter + Python reference checker + Lean checker sources + end-to-end demo | Tier 1 feasibility (rational ops) |
| 1 | `IntervalArith` over all `MathFunc`; feasibility over the transcendental set | Tier 1, full factorable |
| 2 | `Convex.lean` + convex-fast-path emitter | Tier 2 |
| 3 | `LPDuality` + `Covering` + linear/MILP leaves | Tier 3 (MILP) |
| 4 | `Envelopes` McCormick/bilinear | Tier 3 (QP/QCQP) |
| 5 | Remaining `MathFunc` envelopes | Tier 3, full factorable |

## 9. Phase-0 as-built (this slice)

What ships in this change:

1. **Emitter** `python/discopt/certificate/` producing a Tier-1 feasibility certificate
   (`model` + `incumbent`) from a solved `Model`/`SolveResult`.
2. **Python reference checker** `refcheck.py` — the same algorithm the Lean `checkFeasible`
   implements, run in tests to confirm valid certificates are accepted and tampered ones
   rejected. This is the executable oracle for the Lean port.
3. **Lean sources** under `lean/` (`Discopt/Model.lean`, `Discopt/Checker.lean`, a
   `check` executable, `lakefile.lean`, `lean-toolchain`) implementing the Tier-1
   checker over `ℚ` against Lean core (no Mathlib), with `checkFeasible_sound`.
4. **Demo** `scripts/lean_certificate_demo.py` — solve a small NLP + a small MILP, emit
   certificates, run the reference checker (accept), tamper, re-check (reject).
5. **CLI** `discopt solve --emit-certificate` + `discopt cert-check` (in
   `python/discopt/cli.py`), so the emit→check loop is usable from the shell.
6. **Corpus generalization** `python/tests/test_certificate_corpus.py` — certifies real
   MINLPLib instances (`alan`, `nvs03`) end-to-end and confirms a transcendental instance
   (`nvs01`, `sqrt`) is correctly refused.

**Environment note:** in the sandbox this slice was authored in, the org egress policy
blocks GitHub release assets (elan/Lean binaries return 403), so the Lean project could
not be compiled here; it is written to compile against a stock Lean 4 core toolchain in
a developer environment. The Python reference checker provides the runnable end-to-end
verification in the meantime and is byte-for-byte the same decision procedure.

## 10. Verification

- **Lean (developer env):** `lake build` with **zero `sorry`**; `#print axioms
  checkFeasible_sound` shows no `sorryAx`; `lake exe check cert.json` returns `true` on
  demo certificates and `false` on tampered ones.
- **Python (CI / here):** `pytest python/tests/test_certificate.py` — emitter round-trip,
  schema validity, reference checker accepts valid / rejects tampered (bad incumbent,
  inflated objective, integrality violation). Emitter is import-light and touches no
  solver internals, so `node_count` / certified `objective` are unchanged (bound-neutral).
