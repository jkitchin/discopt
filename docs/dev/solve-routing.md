# discopt solve routing — how a problem flows through the solver

Reference map of how `Model.solve()` classifies and dispatches a problem, and how
the two B&B engines (spatial McCormick, NLP-BB) work internally. Line anchors
(`solver.py:NNNN`) are approximate — grep the named symbols if they drift.

Three layers:
1. **Trunk** — entry → selectors → reformulations → classification → per-class dispatch.
2. **Subtree A** — the spatial McCormick Branch-and-Bound engine.
3. **Subtree B** — `_solve_nlp_bb` (nonlinear Branch-and-Bound).

A recurring **soundness invariant** runs through all of it: every *fast* path
(convex QP/MIQP/NLP, MILP simplex, integer-bilinear→MILP) is gated by a
convexity/structure check, and anything not provably safe **falls through to the
spatial McCormick B&B**, which is the sound default. A node bound is only used to
certify optimality (`gap_certified=True`) when it is a *valid* bound; otherwise the
result degrades to `feasible`/`time_limit` with `gap_certified=False`.

---

## 1. Trunk — classification & dispatch

```
Model.solve(...)                                          [modeling/core.py:2680]
  │   knobs: solver, gdp_method, nlp_solver, nlp_bb, gap_tolerance, time_limit,
  │          batch_size, strategy, skip_convex_check, lazy_constraints, tuning, stream
  ▼
solve_model(...)                                          [solver.py:2017]
  │
  ├─ stream=True ───────────────────────────► _solve_streaming  (yields SolveUpdate)
  │
  ├─ EXPLICIT SOLVER SELECTOR
  │     solver="amp" ──► solve_amp           (Adaptive Multivariate Partitioning)
  │     solver="gp"  ──► GP detect → log-space convex reformulation → solve
  │     solver="bb"  ──► force B&B (skip GP auto fast-path)
  │     solver=None  ──► fall through
  │
  ├─ GDP INTERCEPT (model has disjunctions / logic)
  │     gdp_method="oa"    ──► Outer Approximation (solve_oa)
  │     gdp_method="loa"   ──► LOA decomposition
  │     gdp_method="big-m"/"hull" ──► reformulate_gdp → standard MINLP (continue)
  │
  ├─ PRE-DISPATCH REWRITES  (sequential; each may replace `model`)
  │     1. factorable_reformulate   (clear sign-definite denominators / lift factorable terms)
  │     2. integer-bilinear: has_nonconvex_integer_bilinear?
  │           └─ reformulate_integer_bilinear; ADOPT only if result is a *pure MILP*
  │              (classify==MILP AND no residual nonlinear terms; #289 unbounded-factor abort)
  │
  ├─ _pure_continuous?  +  convexity classification (eigenvalue-sound, sense-aware, memoized)
  ▼
classify_problem(model)                                   [problem_classifier.py:54]
  │   decision = f( obj degree {linear|quadratic|higher}, ALL constraints linear?, has int/bin? )
  │   NB: quadratic *constraints* (QCQP) ⇒ NLP/MINLP, not QP/MIQP.
  │
  ├─ LP    ───────────────────────────► _solve_lp        → HiGHS | POUNCE (nlp_solver="pounce")
  │
  ├─ QP    convex?     ──► _solve_qp     (POUNCE; HiGHS-free, #359; JAX-IPM last resort)
  │        indefinite? ──► force_spatial ─────────────────┐ (→ Subtree A)
  │
  ├─ MILP  nlp_solver="simplex"  ──► _solve_milp_simplex  (monolithic Rust B&B; defers on stall)
  │        use_highs & not pounce─► _solve_milp_highs     (HiGHS MIP)
  │        else ─────────────────► _solve_milp_bb         (Rust tree; node-LP = warm simplex | POUNCE-IPM)
  │
  ├─ MIQP  convexity check (eigenvalue):
  │          convex    ──► _solve_miqp_bb  (self-hosted B&B, POUNCE node QPs; HiGHS-free, #359)
  │          nonconvex ──► fall through (→ Subtree A)   (a convex MIQP solver would false-certify)
  │
  └─ NLP / MINLP cascade:                          (ipm/sparse_ipm nlp_solver → POUNCE here)
        ├─ CONVEX NLP fast path (pure_continuous & convex): _solve_continuous
        │     optimal ──► return
        │     not-certified → nonsmooth(abs/min/max) ──► Subtree A (exact piecewise)
        │                     clearable denominator   ──► clear + Subtree A
        │                     else                    ──► return unconverged
        ├─ PURE CONTINUOUS, convexity-unknown / skip_convex_check ──► _solve_continuous
        │     (status="error" ⇒ fall through to Subtree A)
        ├─ CONVEX MINLP (nlp_bb=None & convex & no lazy_constraints) ──► Subtree B (_solve_nlp_bb)
        └─ NONCONVEX MINLP / forced-spatial QP·MIQP / nlp_bb=True   ──► Subtree A
```

| Decision | Criterion | Options |
|---|---|---|
| solver selector | explicit `solver=` | amp / gp / bb / auto |
| GDP intercept | disjunctions + `gdp_method` | oa / loa / big-m / hull |
| reformulations | structural detectors | factorable clear/lift; integer-bilinear→MILP (only if *pure*) |
| classify_problem | obj degree × all-cons-linear × has-int | LP / QP / MILP / MIQP / NLP / MINLP |
| QP·MIQP convexity | eigenvalue test | convex→fast solver; indefinite→spatial (avoids false-optimal) |
| MILP engine | `nlp_solver`, `use_highs` | Rust simplex B&B / HiGHS MIP / Rust tree + POUNCE |
| NLP·MINLP convexity | eigenvalue, memoized | convex→single-NLP or NLP-BB; nonconvex→spatial |
| `nlp_bb` | None(auto) / True / False | auto picks NLP-BB for convex MINLP; True forces it |

---

## 2. Subtree A — Spatial McCormick Branch-and-Bound

Reached for nonconvex MINLP, indefinite QP/MIQP, forced-spatial, and `nlp_bb=True`.
The sound default engine: each node is bounded by a **valid outer relaxation**.

### Root setup (one-time)                                  [solver.py:3350–4130]
```
extract var info ─► root FBBT presolve ─► root infeasible? ─► return INFEASIBLE
   │
   ├─ root OBBT range reduction        [obbt_tighten_root]   gated: has-continuous,
   │     n_vars ≤ 500, NOT known-convex, budget = min(0.1·time_limit, 15s), 3 rounds
   │
   ├─ alphaBB eligibility:  n_vars ≤ 50 & not-convex & evaluator has _obj_fn  → _use_alphabb
   │
   └─ choose per-node BOUND MODE  _mc_mode:
        "auto" ─► has relaxable nonlinearity / branchable?  ─► "lp"
        │                                  pure-discrete fallback ─► "none"
        "lp"  ─► build MccormickLPRelaxer( superposition, psd_cuts, rlt_cuts, rlt_level1 )
        │          RLT resolution:  rlt switch → rlt_level1 (build-time, root bound)
        │                                      + rlt_cuts   (per-node separation)
        │                           auto: engage level1 when n_vars ≤ _AUTO_RLT_LEVEL1_MAX_VARS
        │          no relaxable nonlinearity ─► drop relaxer ─► "none"(pure-discrete) | "nlp"
        │          cuts="auto" & no explicit psd/rlt ─► _apply_auto_cut_policy (RLT vs PSD by structure)
        │          spatial-integer cols ─► tree.set_spatial_integer_cols  (nonlinear-term integers, #194/#202)
        "none"─► alphaBB / interval floor   (rigorous, weak; pure-discrete or no relaxable nl)
        "nlp" ─► node NLP objective         (continuous, nothing branchable)
   + root cut pool: PSD/RLT separated ONCE, inherited (warm) at every node
```

### Node loop (per batch/iteration)                        [solver.py:4130–5660]
```
deadline = t_start + time_limit
while open nodes and not finished:
  export batch from Rust tree  (size = batch_size)
  for each node in batch:
     ├─ deadline passed? ─► skip (bound = -inf, decertify gap), exit at next batch top
     ├─ NODE BOUND (one of):
     │     McCormick LP relaxer.solve_at_node   (backend: Rust simplex | auto/HiGHS;
     │        rebuilds the lifted LP at the box + inherited cuts + on-demand separation)
     │     node NLP                              (root multistart / warm-started child)
     │     alphaBB / interval floor              (_mc_mode="none")
     │     relaxation infeasible ─► RIGOROUS FATHOM (sentinel; subtree pruned soundly)
     ├─ per-node OBBT  (iteration>0, budget, incumbent cutoff)   range reduction on branched nodes
     ├─ cut separation (capped rounds): multilinear / edge-concave / PSD / RLT
     ▼
  PRIMAL HEURISTICS:
     iteration 0  ─► feasibility_pump, fractional_diving        (find first incumbent)
     iteration>0 & gap-open & not-convex ─► LNS layer:
        node-diving (diversify) · RINS (improve) · local-branching (improve, escalating k)
        (sound: re-verify feasible, inject only on strict improvement; dual bound untouched)
  BRANCHING (branching_policy):
     "gnn"  ─► select_branch_variable_gnn
     else   ─► objective-gating priority (_select_priority_branch_var)
               + pseudocost; strong branching (_strong_branch_lp) for unreliable
                 pseudocosts below tree.reliability_threshold()
  import results to tree · node_callback · check termination
TERMINATION: tree finished | gap converged | deadline | node cap
  ─► dual recovery at incumbent ─► gap certification (gap_certified only if bound valid)
```

| Per-node decision | Criterion | Options |
|---|---|---|
| bound mode | relaxable nonlinearity? pure-discrete? convex? | lp (McCormick) / none (alphaBB) / nlp |
| LP backend | relaxer `backend` | Rust warm simplex (default) / auto→HiGHS cross-check |
| cuts | `cuts`, `psd_cuts`, `rlt_cuts`, `rlt` | RLT level-1 + per-node / PSD / multilinear / edge-concave / auto-policy |
| per-node OBBT | iteration>0, budget, incumbent cutoff | on / skipped |
| branching | `branching_policy` | fractional / pseudocost+priority+strong / gnn |
| primal | iteration, gap-open, convexity | feasibility_pump+diving (root) · node-diving/RINS/local-branching (LNS) |

---

## 3. Subtree B — `_solve_nlp_bb` (nonlinear Branch-and-Bound)

Reached for convex MINLP (auto) and `nlp_bb=True`. Each node solves the original
**NLP with discrete vars bound-fixed** instead of a McCormick relaxation. For a
*convex* MINLP the converged node NLP objective is a valid lower bound (certifies
gap); for nonconvex it runs in heuristic mode (`gap_certified=False`).

### Setup                                                  [solver.py:6084–6236]
```
tree.initialize · evaluator = cached_evaluator(model) · infer constraint bounds
warm-start: initial_point feasible? ─► tree.inject_incumbent
_use_pounce_batch = (nlp_solver="pounce" and n_vars ≥ _POUNCE_BATCH_MIN_VARS)
```

### Node loop                                              [solver.py:6236–6660]
```
iteration = 0
while True:
  elapsed ≥ time_limit? ─► break          (deadline)
  export batch from tree
  per-node FBBT infeasibility precheck · optional in-tree presolve (stride)
  ├─ NODE NLP SOLVE:
  │     _use_pounce_batch & n_batch>1 ─► _solve_batch_pounce   (batched KKT, GIL-amortized)
  │     else (serial)                 ─► _solve_node_nlp       (root: multistart; child: warm-start)
  │     node objective = bound (valid for convex ⇒ certifies; heuristic for nonconvex)
  ├─ ROOT PRIMAL (iteration 0):
  │     RENS (primary)        [gated: ≤ max_free=24 fractional integers; else bail]
  │       sub-MINLP fixed to the relaxation's integer rounding, solved exactly (rens=False sub-solve)
  │     feasibility_pump ─► fractional_diving         (fallbacks when RENS bails / no incumbent)
  ├─ import results to tree · node_callback
  ├─ LNS IMPROVEMENT LAYER  (#321; iteration>0, has integers, gap-open):
  │     RINS (between incumbent & node relaxation) · local-branching (Hamming-ball sub-MIP, k∈{2,5,10})
  │     recursion guard: _lns_enabled (the local-branching sub-solve passes False; never nests)
  │     sound: re-verify integer+constraint feasible, inject only on strict improvement
  └─ termination: tree.is_finished() | gap converged | deadline
  iteration += 1
─► dual recovery at incumbent ─► gap certification
```

| `_solve_nlp_bb` decision | Criterion | Options |
|---|---|---|
| node solve path | `nlp_solver="pounce"` & n_vars≥threshold & batch>1 | `_solve_batch_pounce` / serial `_solve_node_nlp` |
| node NLP start | iteration | root multistart / warm-start from parent |
| root primal | RENS gate (≤24 fractional ints) | RENS / feasibility_pump / fractional_diving |
| LNS improvers | iteration>0, integers, gap-open, `_lns_enabled` | RINS + local-branching / skipped |
| bound validity | convexity | valid (certifies) / heuristic (`gap_certified=False`) |

> Note (perf, 2026-06): on the syn/rsyn/clay families that take this path, the LNS
> improvers (#321) close the gap where the incumbent is near a good basin
> (clay0204m 43%→12%) but cannot escape far basins (syn40m) — there the heuristics
> return worse points than the B&B's own node incumbent, so the residual gap is a
> global-*search* problem, not a primal one. See `performance-plan.md`.
