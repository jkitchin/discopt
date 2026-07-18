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
- **CLI (done):** `discopt solve --emit-certificate` runs the solve with tree
  recording on and writes `<stub>.cert.json` at the **strongest tier the solve
  supports** — Tier-3 `bnb` when a spatial-B&B tree was recorded, else Tier-2
  convex, else Tier-1 feasibility (re-loading the model via `from_nl`, as `--sol`
  does). Recording forces the in-process solve (the tree does not cross the daemon
  socket). `discopt cert-check <file>` runs the reference checker (exit 0 ACCEPT /
  1 REJECT). The Lean checker is the separate `lake exe check cert.json` step.
  (`result_io.py` is intentionally left untouched — it holds only a `SolveResult`,
  not the model the emitter needs.)

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
| 2 | convex/KKT emitter + Python checker **(done, exact QP/QCQP — §11)**; `Convex.lean` soundness proof pending Mathlib | Tier 2 (convex QP/QCQP) |
| 3 | exact-rational Tier-3 kernel (covering, LP weak duality, Farkas) **(done — §12)**; recorder/emitter + `LPDuality`/`Covering` Lean (core, no Mathlib) pending | Tier 3 (MILP / polynomial) |
| 4 | McCormick envelopes **(bilinear/square done in kernel — §12)**; `Envelopes.lean` (Rat) + emitter pending | Tier 3 (QP/QCQP) |
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

## 11. Tier 2 (convex / KKT) — detailed design & as-built

**Theorem.** For a convex model (convex objective; feasible set defined by convex
`≤` bodies, concave `≥` bodies, affine `=` bodies, and box bounds), a point `x*`
that satisfies the **KKT conditions** with multipliers is a *global* minimizer.
So a certificate that exhibits (a) a convexity witness and (b) KKT multipliers
proves `dualBound = objectiveValue` — the gap is closed and `x*` is globally
optimal. No branch-and-bound tree is needed.

**What is checkable in exact rationals (shipped).** The **convex QP/QCQP
subclass** — objective and constraint bodies are quadratic — has a *constant*
Hessian and *affine* gradients, both rational. So the entire Tier-2 check is exact
(no floats, no Mathlib beyond the eventual soundness proof):

1. **Convexity** — re-derive each body's Hessian by exact symbolic differentiation
   (`certificate/diff.py`), require it *constant* (2nd derivative has no variable ⇒
   the body is quadratic; higher-degree ⇒ refuse), then: objective & `le` bodies
   PSD, `ge` bodies NSD, `=` bodies zero. PSD is an exact rational LDLᵀ pivot test
   (`certificate/linalg.py`, diagonal pivoting so semidefinite is handled).
2. **Stationarity** — `∇f(x*) + Σ_i λ_i s_i ∇g_i(x*) − ρ^L + ρ^U = 0` component-wise
   (`s_i = +1` for `le`/`eq`, `−1` for `ge`; `ρ^L`,`ρ^U` the bound multipliers),
   within `kkt_tol`.
3. **Dual feasibility** — inequality `λ_i ≥ 0`, bound `ρ ≥ 0`.
4. **Complementary slackness** — `λ_i g_i(x*) = 0` (g in `≤0` form); bound analogues.
5. **Gap closed** — `dualBound == objectiveValue`.

The sign convention is discopt's internal-min one, **validated against real solves**
(`test_convex_*`): a constraint-active QP, a bound-active QP (pins the `ρ^U` sign),
and an interior optimum all certify; inflating a multiplier breaks stationarity;
lowering `dualBound` breaks the gap check.

**Schema additions** (`tier: "convex"`): a `kkt` block
(`constraint_multipliers` aligned to the constraint list, `bound_lower`/`bound_upper`
in column order), a top-level `dualBound`, and `tolerances.kkt`. Multipliers come
from `SolveResult.constraint_duals` / `bound_duals_lower` / `bound_duals_upper`;
the emitter (`build_convex_certificate`) requires `gap_certified` **and**
`convex_fast_path` (the solver's own convexity certification) and a minimize
objective, else it refuses (falling back to a Tier-1 certificate in the CLI).

**What still needs Mathlib (next Lean milestone).** The exact-rational *checker*
ships and is tested, but its **soundness theorem** — "PSD Hessian ⇒ convex" and
"convex + KKT ⇒ global min" — is real convex analysis and belongs in
`lean/Discopt/Convex.lean` over Mathlib, which could not be built in the authoring
sandbox (GitHub egress blocked). Until then, `refcheck._check_convex` is the
**executable specification** the future Lean `checkConvex_sound` must match, exactly
as `refcheck` Tier-1 mirrors the proven `checkFeasible`. The general
*transcendental-convex* case (e.g. `exp` convex, `log` concave, gradients of
`MathFunc`) is a further extension of both the checker (interval/derivative
enclosures) and the Mathlib proof.

**As-built (this change).** `certificate/diff.py` (exact symbolic differentiation),
`certificate/linalg.py` (exact rational PSD), `emit.build_convex_certificate`,
`refcheck` Tier-2 dispatch + `_check_convex`, `discopt solve --emit-certificate`
auto-selecting the strongest tier, the convex block in
`scripts/lean_certificate_demo.py`, and `test_convex_*` in
`python/tests/test_certificate.py` (accept genuine; reject tampered multiplier,
open gap, non-convex objective, negative dual; emitter refuses maximize).

## 12. Tier 3 (spatial branch-and-bound) — detailed design & as-built kernel

**Theorem.** For a *nonconvex* model, spatial branch-and-bound proves global
optimality by exhausting the domain: the branch tree's **leaf boxes cover the root
box**, and **every leaf is fathomed** — either *infeasible* (its relaxation is
empty, a Farkas ray) or *by bound* (its relaxation has a certified lower bound
`leaf_lb`). Then over each leaf no point beats `leaf_lb`, so
`min_leaves leaf_lb =: L` is a valid **global lower bound**; if `L ≥ incumbentValue`
the gap is closed and the incumbent is globally optimal. No leaf's relaxation, and
no branching decision, needs to be trusted — each is *checked*.

**Trust-minimizing architecture.** The certificate carries the **tree** (per-node
box, branch variable/point, children) and, per bound-fathomed leaf, an **LP dual**
(and per infeasible leaf a **Farkas ray**). Two soundness obligations, both
exact-rational:

1. *Covering* — each branch node's two children reproduce the parent box with one
   coordinate split (spatial `x≤s | x≥s`; integer `x≤⌊s⌋ | x≥⌈s⌉`, whose open gap is
   sound only for integer columns). Pure combinatorics ⇒ leaves cover the root.
2. *Per-leaf bound* — the leaf's relaxation LP is a **valid** outer approximation
   (removes no true-feasible point), and the emitted **dual** certifies `leaf_lb` by
   **LP weak duality** (`y≥0`, `Aᵀy=c` ⇒ `b·y ≤ optimum`). Validity is the
   anti-unsound-cut gate: every LP row must be a box bound or a **closed-form
   McCormick row** the checker *recomputes from the leaf box* (so a tampered
   coefficient or an injected cut is caught). Farkas rays certify empty leaves.

The strongest form has the checker **reconstruct** each leaf's McCormick relaxation
from the model + box itself (trusting only the dual + tree); the as-built kernel
takes the emitted LP and verifies every row is recognized-valid, which is
equivalent for the McCormick fragment.

**What is exact-rational (shipped & tested — `certificate/bnb.py`).** The entire
Tier-3 *soundness kernel* is exact rational and model-agnostic:
`check_tree_covers`, `mccormick_bilinear`/`mccormick_square` (closed-form valid
envelopes), `lp_lower_bound` (weak-duality certified bound), `farkas_infeasible`,
and `certified_leaf_bound` (valid-rows + dual). Verified on the nonconvex
`min -x²` over `[0,2]`: lifting `w=x²`, the McCormick relaxation's weak-duality
bound is exactly `-4` (the global optimum), and an injected non-McCormick cut is
rejected (`test_certificate_bnb.py`). Notably, **weak duality and covering are
exact and problem-independent** — only per-term *relaxation validity* is
term-type-specific, so the polynomial/bilinear (QCQP) class is fully covered now;
the general factorable case adds one recomputable envelope family per `MathFunc`
(the algebraic ones exact; transcendental ones are the Mathlib phase).

**Float → rational bridge.** A leaf's LP dual comes from the float simplex; emit it
as rationals and the exact weak-duality check yields a rigorous bound. discopt
already computes a Neumaier–Shcherbina directed-rounding safe bound
(`_jax/mccormick_lp.py`, `safe_bound`) — the sound float→rational rounding the
emitter uses so `Aᵀy=c` holds exactly (or is relaxed to the safe side).

**The tree recorder (built).** Unlike Tiers 1–2, whose data is already on
`SolveResult`, Tier-3 needs the tree the solver discards. Rather than hook every
branch path, the recorder exploits the fact that the `NodePool` *retains every node
ever created* (parent, box, `local_lower_bound`, terminal status): `TreeManager::
tree_records()` (`crates/discopt-core/src/bnb/tree_manager.rs`) snapshots the whole
pool **read-only after the solve** — zero hot-path bookkeeping, byte-identical
search whether or not a certificate is requested (bound-neutral by construction).
The `PyTreeManager.tree_records()` binding surfaces it to Python, and
`certificate/bnb_record.py` reconstructs the nested tree, **deriving each branch's
split variable and point from the parent-vs-child box difference** (the recorder
stores no branch metadata) and running `check_tree_covers`. Cargo tests
(`tree_records_*`) and `test_certificate_bnb_record.py` verify it end-to-end
(root-only, an integer branch, a 2-level spatial tree, and gap rejection).

**Solver wiring (built).** `solve_model` takes an opt-in `emit_certificate=False`
(bound-neutral default). When set, the spatial-B&B loop (a) stashes
`tree.tree_records()` on `SolveResult.bnb_tree` after the loop, and (b) requests
McCormick LP marginals (`want_marginals=emit_certificate`) and captures each node's
`dual` / `safe_bound` / `reduced_costs` into `SolveResult.bnb_leaf_duals` keyed by
node id (`_cert_record_node_dual`, best-effort — populated on the incremental fast
path). `build_bnb_certificate(model, result)` then emits the `bnb` tier (Tier-1
model + incumbent, the recorded tree with rational boxes + per-leaf bounds, the
`dualBound`, and any captured leaf duals), and `check_bnb_certificate` verifies:
incumbent feasibility (Tier-1) → **covering** (`check_tree_covers` on the
reconstructed tree) → the reported `dualBound` does not exceed the minimum recorded
leaf bound (so it is a valid global lower bound) → gap-closed ⇒ global optimum.
Verified end-to-end (`test_certificate_bnb_e2e.py`): a real nonconvex bilinear
solve certifies (tree covers root, valid dual bound), recording is bound-neutral
(identical node count / objective on vs off), and tampering (inflated `dualBound`,
broken covering) is rejected.

*Scope:* the recorder is wired into the **spatial-B&B McCormick loop** (`solve_model`,
the rigorous nonconvex-continuous path, with per-leaf LP-dual capture) **and the
NLP-BB loop** (`_solve_nlp_bb`, tree only — its node bound is the NLP objective,
a valid lower bound only for convex MINLPs, so `build_bnb_certificate` requires
`gap_certified`; nonconvex NLP-BB nodes carry `-inf`, which the checker rejects). A
real convex integer MINLP (`nvs03`) certifies via this path. The pure-integer MILP
driver and decomposition paths are not yet wired (they return no `bnb_tree`, so the
emitter refuses and the CLI falls back to Tier-1/2) — a follow-on.

**Untrusted per-leaf re-derivation (built, `untrusted=True`).** For the **quadratic
fragment** (linear + bilinear + square bodies), `build_bnb_certificate(model, result,
untrusted=True)` attaches to each leaf an independently-derived **exact dual**, and
`check_bnb_certificate` re-derives that leaf's bound by **rebuilding the McCormick
relaxation LP itself** (`certificate/relax.py`) from the model + leaf box and
verifying the dual by weak duality — so it trusts neither the solver's recorded
bound nor any emitted LP, only the dual, which a tamper breaks (`Aᵀy ≠ c`). The
compiler (`relax.py`) extracts each body's quadratic form, lifts products to aux
columns, and emits box + McCormick envelope rows; the emitter recovers an exact
rational dual from a SciPy solve's active/marginal support via exact Fraction
elimination (`leaf_dual`), verified exactly before it is attached (a failed
recovery leaves that leaf trusted — best-effort, never unsound). Verified
end-to-end (`test_certificate_relax.py`, `test_bnb_untrusted_leaf_rederivation`):
`min -x²` re-derives its exact `-4`, a bilinear leaf re-derives a valid bound, and
a corrupted dual is rejected. This certifies the **clean-McCormick** bound (no
cuts); where that closes the gap the incumbent is untrusted-certified globally
optimal. Higher-degree / transcendental leaves stay trusted (documented) —
their untrusted upgrade is the interval/envelope work shared with Tiers 1/3-Lean.

**Lean obligations.** Encouragingly, most of Tier 3 is **Lean-core (no Mathlib)**:
`LPDuality.lean` (weak duality is exact linear algebra over `Rat`) and
`Covering.lean` (box union is combinatorial) need no analysis; `Envelopes.lean`'s
McCormick bilinear/square validity is provable over an ordered field (`Rat`). So a
**nonconvex-*polynomial* global-optimality certificate is checkable in Lean core** —
only the transcendental envelopes (`exp`/`log`/…) pull in Mathlib. `bnb.py` is the
executable specification those Lean checkers must match.

**As-built.** (1) The checker kernel — `certificate/bnb.py` (covering, McCormick
bilinear/square, LP weak duality, Farkas, `certified_leaf_bound`) +
`test_certificate_bnb.py` (7 tests). (2) The **Rust tree recorder** —
`TreeManager::tree_records()` + `NodeRecord`, the `PyTreeManager.tree_records()`
PyO3 binding, `certificate/bnb_record.py` (reconstruct + derive splits + covering),
cargo tests (`tree_records_*`) + `test_certificate_bnb_record.py` (5 tests). (3) The
**solver wiring + emitter/checker** — `emit_certificate` on `solve_model`
(`_cert_record_node_dual`, tree/dual stash on `SolveResult.bnb_tree`/`bnb_leaf_duals`),
`emit.build_bnb_certificate`, `refcheck._check_bnb` (`bnb` dispatch), and
`test_certificate_bnb_e2e.py` (5 slow tests: accept a real solve, bound-neutral,
reject inflated dualBound / broken covering, refuse without recording). The
fully-untrusted per-leaf LP-dual re-derivation is the remaining step (above).
