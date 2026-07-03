# Solver Core Review — Soundness, Correctness, SOTA

**Date:** 2026-07-03
**Scope:** the B&B solver core proper — `python/discopt/solver.py` (the
orchestrator, per-node bounding, convex fast path, alphaBB node bound), the
convexity certifier (`python/discopt/convexity/`), the JAX relaxation compiler and
McCormick primitives (`python/discopt/_jax/`), factorable/GDP reformulation
(`gdp_reformulate.py`, `factorable_reform.py`), the cutting-plane and MILP
relaxation layer (`cutting_planes.py`, `rlt_cuts.py`, `psd_cuts.py`, `soc_cuts.py`,
`edge_concave.py`, `milp_relaxation.py`), the outer-approximation paths
(`oa.py`, `gdpopt_loa.py`), the NLP evaluator (`nlp_evaluator.py`), and the Rust
per-node engine touchpoints (`crates/discopt-core/src/lp/`, `presolve/fbbt.rs`,
`crates/discopt-python/`).
**Method:** six delegated verification agents, each reading its load-bearing files
end to end and then building **numerical repros** — deterministic false-certificate
constructions where a bug was suspected, and adversarial fuzz (4,000-trial
certificate fuzz with numerical-Hessian cross-check; composite-envelope containment
fuzz; ~400 differential decomposition instances; primitive soundness sweeps) where
soundness was claimed. Nothing in this document is asserted without a run.

A global solver's product is its **certificate** (CLAUDE.md §1). This review's
question was narrow and adversarial: *can the core be made to return
`gap_certified=True` on a wrong answer, or emit an invalid dual bound / invalid
cut?* The headline is two-sided.

## Bottom line

**The highest-stakes gate is sound.** The convexity certifier and the convex
fast-path it feeds — the single largest lever on false certificates, because a
false-convex verdict short-circuits spatial B&B into one NLP solve — **could not be
tricked** across 901 non-abstaining certificates under randomized fuzz (every
`CONVEX`/`CONCAVE` verdict cross-checked against the sampled Hessian spectrum; zero
false). The primal/dual simplex, the B&B tree manager, the PyO3 boundary, the
Gurobi/AMP nonconvex gating, and the RLT/PSD/SOC/edge-concave cut families are
likewise verified sound.

**But the default solve path has two new demonstrated false-optimals** (SC-1,
FR-1), one previously-open P0 is now **confirmed with a deterministic repro**
(C-17), the C-31 FBBT array-collapse bug is shown to reach the **certified LP dual
bound** (broader than its card stated), and C-23's "harmless" label is **falsified**
(DIV-1). These are the actionable output of this pass.

---

## 1. New / confirmed false-certificate findings

> **Backlog IDs (`correctness-issues.md`).** SC-1 = **C-33**, FR-1 = **C-34**,
> OA-1 = **C-35** (§2); DIV-1 escalates existing **C-23** (P3→P1); C-17 and the
> FBBT collapse **C-31** are existing/renumbered cards confirmed/broadened here.
> (The C-25..C-28 range is owned by the NN backlog; the four Fable-review core P0s —
> CORE-1/CORE-2/TG-1/NM-1 — are carded as C-29..C-32.)

| ID | Severity | Loc | Finding | Status |
|----|----------|-----|---------|--------|
| **SC-1 (C-33)** | **P0 (default path)** | `solver.py:3716-3738` (fallback), cert at `:6889-6896` | **Pure-continuous fallback certifies a nonconvex model.** When `skip_convex_check or not _pure_continuous_convexity_known`, a pure-continuous model is routed to a single NLP solve whose local optimum is emitted **with `gap_certified=True`**. | VERIFIED — nonconvex double-well returns obj −37, bound −37, `gap_certified=True`; true min −64 |
| **FR-1 (C-34)** | **P0 (default path)** | `gdp_reformulate.py` (`**` arm) | **Even-power bound over a zero-straddling base uses endpoint-only bounds.** Only `p==2` got the straddle-aware `[0, max(lb²,ub²)]`; `p≥4` over a base whose box straddles 0 got `[min(endpoints), max(endpoints)]`, omitting the interior min at 0. `_bound_expression(x**4)` on `[−2,2]` → `[16,16]` (true `[0,16]`). Feeds the aux box (`factorable_reform.py:254`) and denominator clearing (`:707`). | **FIXED (PR #425)** — straddle case now generalized to all even integer powers → `[0, max(lb^p,ub^p)]`. Was VERIFIED e2e: `min (x−0.5)²+(y−1)² s.t. x**4·y ≤ 1` returned a false certified optimum (obj≈3.13 for `x,y∈[−2,2]`); now returns ≈0 at x=0.5. |
| **C-17** | **P0 (open → confirmed)** | `solver.py:4462-4466`, `:551/:556-558`, `:4974/:5228`; `alphabb.py:56` | **alphaBB node bound is unsound.** α comes from `estimate_alpha` (Hessian sampled at 100 fixed-PRNG points), and `_compute_alphabb_bound` checks convexity of `L = f − Σα(x−lb)(ub−x)` **only at the box center**. A negative-curvature band narrower than the sample spacing → α=0, center Hessian passes the gate, bound is returned as valid. | VERIFIED — deterministic repro below; false optimal |
| **DIV-1** | **P1 (falsifies C-23)** | `_jax/mccormick.py:119-135` (`relax_div`), wired `relaxation_compiler.py:719-726` | **Invalid convex underestimator for nonlinear denominators.** `relax_div` evaluates the reciprocal at `mid_r = ½(cv_r+cc_r)`; when the denominator is nonlinear `cv_r ≠ cc_r` even at a point, so `1/mid_r` sits **above** the true `1/(·)` → `cv > f`. C-23 declared this "harmless" having tested only variable/affine denominators (which point-collapse to exact). | VERIFIED — `1/(x*y)` on `[0.3,2]×[0.4,1.8]` at (1,1): cv=1.334 > true 1.0; cv > f at 3000/3000 sampled pts |
| **C-31 (broadened)** | **P0 (open, wider blast radius)** | `tightening.py:114-121` + `_fbbt_argument_box` `milp_relaxation.py:4338-4352`; Rust `fbbt.rs:1204-1208,1105-1111` | **FBBT array-block collapse reaches the certified LP dual bound.** The univariate-rescue path builds a McCormick envelope over `fbbt_box`, which stamps one per-block interval (Rust seeds from element-0 bounds) onto every scalar slot. On heterogeneous per-element array bounds the box **excludes feasible points** → invalid envelope in the certified LP relaxation, not merely invalid conflict cuts (its documented reach). | VERIFIED — Rust characterization tests pin the collapse; consumer chain traced to certified bound |

### C-17 deterministic repro

`f(x) = ½x² − B·exp(−(x−a)²/2s²)` on `[−2,2]`, B=4, a=1 (root node, so `estimate_alpha`
and `_compute_alphabb_bound` see the same box):

| s (spike width) | sampled α | needed α (true) | H(center) | alphaBB "bound" | true box min | invalid? |
|---|---|---|---|---|---|---|
| 0.006  | **0.000** | 24,791  | 1.0 (passes gate) | **0.000** | −3.500 | **+3.5** |
| 0.003  | 0.000 | 99,168  | 1.0 | 0.000 | −3.500 | yes |
| 0.0015 | 0.000 | 396,674 | 1.0 | 0.000 | −3.500 | yes |

The narrow negative-curvature band falls between every deterministic sample point → α=0;
the center Hessian (=1.0) passes the PSD gate; the function returns **0.0 as a valid lower
bound while the true minimum is −3.5**. In B&B this fathoms the node holding the true
optimum → wrong answer certified. Fix vehicle `rigorous_alpha` (`alphabb.py:271`,
interval-Gershgorin) already exists and is correct — the bug is that the sampled/center
path is used instead.

### Reachability summary

- **SC-1, FR-1** are on the **default** solve path (no opt-in flag) — these are the two
  most urgent items in this pass.
- **C-17** fires whenever `_alphabb_eligible` (`n_vars ≤ 50 and not convex`,
  `solver.py:4057`) and `_mc_lp_relaxer is None` — the default for small nonconvex
  models where the Rust LP relaxer is not the bound source.
- **C-31** now reaches the certified LP dual bound via `_fbbt_argument_box`; still
  requires a model with heterogeneous per-element array bounds (X-2 class).
- **DIV-1** writes the invalid value into `result_lbs[i]` under opt-in
  `mccormick_bounds="nlp"` only; the default `auto` mode routes to the Rust
  `MccormickLPRelaxer`, which does not use JAX `relax_div`. The #120 decertification is
  currently the *only* backstop — it should be fixed at the math level, not left to a
  downstream guard (CLAUDE.md §3).

---

## 2. Secondary findings (latent / opt-in)

| ID | Severity | Loc | Finding |
|----|----------|-----|---------|
| FR-2 | P1 latent | `nlp_evaluator.py:199-201` | Builds from `model._constraints` only — blind to builder-resident rows (X-1 class). Latent on the default path (saved by the Rust representation) but **live** for `nlp_ipopt.py:254` and the examiner. |
| OA-1 (C-35) | P1 opt-in | `oa.py:209-236,893-931,916`; `gdpopt_loa.py:314-370,225` | Non-rigorous NLP failure → **unconditional** no-good cut → possible false infeasible/optimal. Opt-in OA/LOA path only. |
| Rust-1 | P3 latent | `lp/batch.rs:68-70` | `solve_lp_batch` does not unscale the returned dual/ray; caller currently discards them, so latent. |
| Rust-2 | P3 robustness | `crates/discopt-python/…expr_bindings.rs:372,379` | Panics on non-contiguous numpy input instead of raising a Python error. |

---

## 3. Verified SOUND (the positive results — do not rebuild)

These are the load-bearing confirmations. Each was adversarially probed, not assumed.

- **Convexity certifier (`convexity/`).** `certify_convex`/`classify_model` never
  certified a nonconvex function convex: every hand-built probe (`x*y`, `x²−y²`,
  indefinite `x²+y²−3xy`, `exp(−x²)`, `sin`, `x³`, `x/y`) abstained; the PSD case
  `(x−y)²` → CONVEX. **901 non-abstaining certificates in 4,000-trial fuzz, 0 false**
  (each cross-checked against the numerically-sampled Hessian spectrum). The
  interval-Gershgorin eigenvalue layer is rigorous (Higham summation-error inflation,
  outward rounding, `±inf` on non-finite Hessian). The interval-Hessian AD **correctly
  seeds the Variable leaf from the box** (`interval_ad.py:476-542`) — the certifier's
  analog of the numpy NM-2 box-drop bug is **not** present here. DCP composition and the
  atom table are correct; notably `asin`/`acos` curvature in the **classifier** is
  *correct* (`lattice.py:414-439`) — the C-32/NM-1 inversion is confined to the McCormick
  *primitive*, not the DCP classifier. Pattern recognizers that assert CONVEX directly
  (quad-over-linear, geo-mean, perspective, norm2, softplus, `1/x`) are Jensen-clean.

- **Convex fast path fires only on a rigorous verdict.** `solver.py:3651` gates the
  single-NLP fast path on `_pure_continuous_convexity_known and _pure_continuous_is_convex`;
  the MIQP path (`:3619`) on `_root_convexity_known and _root_is_convex`.
  `skip_convex_check` only *downgrades* certification (`:6959` emits an uncertified
  warning) — it never fabricates one. `convex_fast_path=True` is telemetry, not the
  certificate (a nonconvex e2e returned `convex_fast_path=True` but `gap_certified=False`).
  **(SC-1 is the exception: the *pure-continuous fallback* at `:3716` is a different path
  that does emit the certificate — that is the bug, not this gate.)**

- **JAX relaxation compiler Variable leaf is correct** (`relaxation_compiler.py:403-421`):
  returns `(x_cv[offset], x_cc[offset])`; box enters through the call convention. Broad
  composite containment fuzz (`exp(x*y)`, `(x*y)²`, `log(x*y)`, `x*y*z*w`, `sin(x*y)`,
  `(x+y)³`, `tanh(x*y)`) — **0 violations**. (Contrast: the *numpy* compiler drops the box
  — NM-2 — but is compiled-but-unused.)

- **Cutting planes.** RLT McCormick rows (`cutting_planes.generate_rlt_cuts`), RLT
  constraint×bound (`rlt_cuts.py` linear + quadratic), PSD/moment (`psd_cuts.py`,
  `vᵀM(x,X)v ≥ 0` globally valid), SOC (`soc_cuts.py`, Cauchy-Schwarz), edge-concave
  (`edge_concave.py`, intercept recomputed over box vertices — valid by Tardella), and the
  **rigorous** alphaBB quadratic OA (`cutting_planes.py:534-609`, exact constant-Hessian
  eigenvalue + safety) — all verified term-by-term for sign/degree; box-dependent cuts are
  separated at the **root box** and inherited read-only, so no sub-box pool leak. OA
  tangents emitted only for `convex_mask=True` rows.

- **Primal/dual simplex, tree manager, PyO3 boundary** (from the extraction review):
  bound arithmetic, incumbent/best-bound bookkeeping, and the zero-copy numpy marshaling
  verified — no invalid global bound, no incumbent corruption.

- **Gurobi/AMP nonconvex gating, decomposition layer** (~400 differential instances, no
  false certificate): complete-dual Benders cuts, GBD projected multipliers, Lagrangian
  rigorous-subproblem bound, node-bounder `max()` combination — all valid.

---

## 4. SOTA notes

The convexity-certification stack (interval-Gershgorin + DCP lattice + pattern
recognizers, with rigorous outward rounding) is **at or above** the correctness bar of
the certified-relaxation machinery in comparable global solvers — the rigorous outward
rounding and the box-seeded interval Hessian are a genuine hardening. The
false-certificate defects found here are **not** in that stack; they are in older
node-bound paths (sampled alphaBB, the pure-continuous fallback) that predate it and
should route through the rigorous machinery that already exists (`rigorous_alpha`, the
certifier gate). Fixing SC-1/FR-1/C-17 is convergence on the solver's own best
components, not new research.

---

## 5. Plan (for Opus)

Priority order (all P0 first, default-path before opt-in):

1. **`fix(correctness): SC-1`** — the pure-continuous fallback (`solver.py:3716-3738`)
   must **withhold** `gap_certified` unless `_pure_continuous_convexity_known and
   _pure_continuous_is_convex`. When convexity is unknown, fall through to spatial B&B,
   not a certified single NLP. **Acceptance:** the double-well repro returns
   `gap_certified=False` (or the true −64 via B&B); no node-count/objective drift on the
   certifying panel for models that were *already* convex.

2. **`fix(correctness): FR-1`** — the `**` arm (`gdp_reformulate.py:493-506`) must, for
   even `p` over a zero-straddling base, return `[0, max(lb**p, ub**p)]` (the interior
   min at 0), matching the `p==2` branch generally. **Acceptance:** `_bound_expression(x**4)`
   on `[−2,2]` → `[0,16]`; the `x**4·y ≤ 1` repro returns 0 at x=0.5.

3. **`fix(correctness): C-17`** — route the alphaBB node bound through `rigorous_alpha`
   (`alphabb.py:271`) and a whole-box (not center-only) convexity check; or gate the
   sampled path off entirely for certified bounds. **Acceptance:** the spike repro returns
   a bound ≤ −3.5 (valid) or abstains; add the repro as a regression.

4. **`fix(correctness): C-31`** — fix `fbbt_box`/Rust FBBT seeding to carry per-element
   array bounds (X-2 class). Cross-link the `_fbbt_argument_box` consumer into the C-31
   card. **Acceptance:** the heterogeneous-block characterization tests flip from
   "asserts collapse" to "preserves per-element bounds"; the univariate-rescue envelope
   contains all feasible arguments.

5. **`fix(correctness): DIV-1`** — `relax_div`/`_relax_reciprocal` must apply the
   monotone-composition rule over the inner interval: for `1/y`, `y∈[cv_r,cc_r]>0`
   (convex, decreasing), the valid convex underestimator is `1/cc_r`, not `1/mid_r`. Fold
   into C-23 (falsified "harmless"). **Acceptance:** extend the relaxation harness with
   reciprocal-/division-of-nonlinear-inner (currently omitted — CI is blind); the `1/(x*y)`
   repro flips to `cv ≤ f`.

6. Secondary (separate PRs): FR-2 (build NLP evaluator from resident rows too — X-1),
   OA-1 (make the no-good cut conditional on a rigorous infeasibility proof), Rust-1
   (unscale dual/ray), Rust-2 (raise instead of panic on non-contiguous numpy).

Each fix ships with an adversarial regression that **fails before and passes after**
(CLAUDE.md workflow), and the bound-changing ones (C-17, C-31, DIV-1) additionally with
the differential bound test (new bound ≥ old AND ≤ true box optimum) plus feasible-point
sampling.
