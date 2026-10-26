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
  ├─ stream=True ───────────────────────────► _solve_streaming  (NotImplementedError:
  │                                             no backend yields SolveUpdate yet)
  │
  ├─ solver="pounce" ──► _solve_pounce_route  (#1533; before ALL presolve/reformulation)
  │     refuses int/bin vars, GDP/logical constraints, feasibility callbacks
  │     LP  ──► convex_ipm_pounce.solve_lp  (POUNCE lp-ipm)  → "optimal"
  │     QP  ──► certify_psd (exact rational LDLᵀ) ──► convex_ipm_pounce.solve_qp
  │           (POUNCE qp-ipm) → "optimal";  not proved PSD ──► NLP arm
  │     else──► _solve_continuous(tighten_bounds=False)  (filter line-search IPM, once)
  │           → "local_optimal" / "local_limit" / "local_infeasible", never a bound
  │
  ├─ EXPLICIT SOLVER SELECTOR
  │     solver="amp"      ──► solve_amp      (Adaptive Multivariate Partitioning)
  │     solver="gp"       ──► GP detect → log-space convex reformulation → solve
  │     solver="gp-minlp" ──► GP-MINLP detect → y-space node relaxations + integer B&B
  │     solver="bb"       ──► force B&B (skip GP auto fast-paths)
  │     solver=None       ──► CONVEX-MINLP AUTO-ROUTE (#1059; DISCOPT_CONVEX_MINLP_ROUTE, default ON,
  │                             opt-out =0): class ∈ {MIQP, MIQCP, MIQCQP, MINLP} AND certified
  │                             convex at the root AND no option the MIP-NLP family ignores
  │                             (nlp_bb, node_callback, ...) ──► solver="mip-nlp",
  │                             mip_nlp_method="lp_nlp_bb" (HiGHS master; #1658,
  │                             DISCOPT_CONVEX_ROUTE_METHOD=oa restores "oa"); else fall through
  │
  ├─ GDP INTERCEPT (model has disjunctions / logic)          [two-axis contract → §4]
  │     gdp_method="oa"    ──► Outer Approximation (solve_oa) (deprecated → mip-nlp/oa)
  │     gdp_method="loa"   ──► LOA decomposition   (native disjunctive axis)
  │     gdp_method="gloa"  ──► global LOA (relaxation master; native axis)
  │     gdp_method="big-m"/"hull" ──► reformulate_gdp → standard MINLP (continue)
  │     NB: native gdp_method (loa) + solver="mip-nlp" ──► ValueError (contradictory; §4)
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
  ├─ LP    default ───────────────────► _solve_lp_highs  (verified HiGHS route, #1229; lp-milp-highs-routing-plan.md §3)
  │        DISCOPT_LP_MILP_BACKEND=rust► _solve_lp        → Rust simplex | POUNCE (nlp_solver="pounce")
  │        milp_backend="native" ──────► _solve_lp        (same engine; route "native-lp", #1535)
  │
  ├─ QP    convex?     ──► _solve_qp     (POUNCE; HiGHS-free, #359; JAX-IPM last resort)
  │        indefinite? ──► force_spatial ─────────────────┐ (→ Subtree A)
  │
  ├─ MILP  default (exactly linear, no callbacks / lagrangian_bound / nlp_bb=True)
  │                              ──► _solve_milp_highs     (HiGHS MIP, verified; discopt root presolve skipped)
  │        DISCOPT_LP_MILP_BACKEND=rust, or nlp_solver="simplex":
  │                              ──► _solve_milp_simplex  (monolithic Rust B&B; defers on stall)
  │        else ─────────────────► _solve_milp_bb         (Rust tree; node-LP = warm simplex | POUNCE-IPM)
  │        milp_backend="native" (#1535) ► _solve_milp_bb directly — skips HiGHS AND the monolithic
  │                                   engine; honours strategy, batch_size, max_nodes, node_callback,
  │                                   branching_rule (pseudocost | most_fractional | least_fractional
  │                                   | strong, via tree branch hints), milp_cuts=False (no root cut
  │                                   loop); route "native-milp".
  │        node_callback (no milp_backend) ► _solve_milp_bb (HiGHS has no tree); route "milp-tree".
  │        lazy_constraints / incumbent_callback / cut_callback → Subtree A (#748; cut_callback
  │                                   since #1535 — no MILP engine has a cut hook); with native,
  │                                   route "native-spatial". The convex NLP-BB auto-select also
  │                                   steps aside for cut_callback; nlp_bb=True + cut_callback raises.
  │
  ├─ MIQP  convexity check (eigenvalue):
  │          convex    ──► normally never reached: the convex-MINLP auto-route above already
  │                         sent it to mip-nlp/lp_nlp_bb. Reached only when that route refused
  │                         (opt-out =0, or a caller option it ignores) ──► _solve_miqp_bb
  │                         (self-hosted B&B, POUNCE node QPs; #359)
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
        ├─ CONVEX MINLP (nlp_bb=None & convex & no lazy/incumbent/cut callback) ──► Subtree B
        │     (_solve_nlp_bb). As for convex MIQP, only reached when the auto-route above
        │     refused; by default a root-certified convex MINLP is solved by mip-nlp/lp_nlp_bb.
        └─ NONCONVEX MINLP / forced-spatial QP·MIQP / nlp_bb=True   ──► Subtree A
```

| Decision | Criterion | Options |
|---|---|---|
| solver selector | explicit `solver=` | amp / gp / gp-minlp / bb / pounce / auto |
| GDP intercept | disjunctions + `gdp_method` | oa / loa / big-m / hull |
| reformulations | structural detectors | factorable clear/lift; integer-bilinear→MILP (only if *pure*) |
| classify_problem | obj degree × all-cons-linear × has-int | LP / QP / MILP / MIQP / NLP / MINLP |
| QP·MIQP convexity | eigenvalue test | convex→fast solver; indefinite→spatial (avoids false-optimal) |
| LP / MILP engine | `milp_backend=` keyword (`highs`/`native`, wins), else `DISCOPT_LP_MILP_BACKEND` (default `highs`, opt-out `rust`), `nlp_solver` | HiGHS (verified route) / Rust simplex B&B / Rust tree + POUNCE |
| NLP·MINLP convexity | eigenvalue, memoized | convex→single-NLP or NLP-BB; nonconvex→spatial |
| convex-MINLP auto-route | `solver=None`, class MIQP/MIQCP/MIQCQP/MINLP, root-certified convex, `DISCOPT_CONVEX_MINLP_ROUTE` (default on) | mip-nlp/lp_nlp_bb (HiGHS master; `DISCOPT_CONVEX_ROUTE_METHOD=oa` for OA, #1658); refused → NLP-BB / MIQP-BB below |
| `nlp_bb` | None(auto) / True / False | auto: convex MINLP → OA auto-route, NLP-BB only if that refuses; True forces NLP-BB (and blocks the auto-route) |

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
        "lp"  ─► build MccormickLPRelaxer( psd_cuts, rlt_cuts, rlt_level1 )
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
  BRANCHING:
     objective-gating priority (_select_priority_branch_var)
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
| branching | (built-in) | pseudocost + priority + strong branching |
| primal | iteration, gap-open, convexity | feasibility_pump+diving (root) · node-diving/RINS/local-branching (LNS) |

---

## 3. Subtree B — `_solve_nlp_bb` (nonlinear Branch-and-Bound)

Reached for `nlp_bb=True`, and for a convex MINLP under `nlp_bb=None` only when the convex-MINLP auto-route to mip-nlp/lp_nlp_bb refused it (§1). Each node solves the original
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

---

## 4. Two-axis solver API contract (locked, #323)

The solver API has **two orthogonal axes**. They answer different questions and
must never be aliased together. This contract was agreed while reviewing #319/#320
and locked by #323 so later work does not drift; the enforcement below ships and is
tested — do not weaken it without re-litigating the contract.

**Axis 1 — `gdp_method`: *how a disjunctive model is handled.*** Either reformulate
the disjunctions into an algebraic MIP/MINLP, or solve the disjunctive form
*natively* via a logic-based method.

| `gdp_method` | Meaning | Kind |
|---|---|---|
| `big-m`, `hull`, `mbigm`, `auto`, `simplex`, `binary_multiplication` | reformulate disjunctions → algebraic MIP/MINLP | reformulation |
| `loa` | logic-based OA, solved natively on the disjunctive form (`solve_gdpopt_loa`) | native disjunctive |
| `gloa` | global logic-based OA (#1775 G4.3): LOA with a MILP-relaxation master of every row and global subproblems, so its bound needs no convexity certificate (`solve_gdpopt_loa(algorithm="gloa")`) | native disjunctive |
| `ric` | relaxation with integer cuts (#1775 G4.2): the lowered model's linear rows plus no-goods as the master, global subproblems, a no-good only on a certificate (`solve_gdpopt_loa(algorithm="ric")`) | native disjunctive |
| `enumerate` | every configuration solved globally (#1775 G4.2); refuses above `max_configurations` (default 10⁴) instead of truncating; the LB is the min over configurations, published once all have one | native disjunctive |
| `lbb` | logic-based branch and bound (#1775 G4.4 stage 1): best-first over disjunctions; a node is the work model with its partial selection applied, solved by discopt under a per-node budget, so a certified node solve closes its subtree; logic propagation prunes refuted children | native disjunctive |
| `ldsda` | logic-based discrete-steepest descent (#1775 G4.5): a **local** neighbor/line search over external variables (`exactly(1)` groups of Booleans); each point's Booleans are pinned and the rest solved — by the local NLP when the logic decides every disjunction, else by discopt; no bound, never `optimal` | native disjunctive (local) |
| `oa` | *deprecated* — was "OA solver + big-M reform"; now warns and reforms as `big-m` | deprecated |

The native set is `solver.NATIVE_GDP_METHODS`; the subset that runs is
`IMPLEMENTED_NATIVE_GDP_METHODS`. Native methods share one loop
(`discopt.gdp.algorithms.GDPAlgorithm`: deadline into every subsolve, the
§5.4 status rules of #1775 — never `optimal` with a configuration unresolved or
an LB the method cannot certify) and take their options through **`gdp_options`**,
a mapping with Pyomo GDPopt's key names (`iterlim`, `init_algorithm`,
`set_cover_iterlim`, `OA_penalty_factor`, `max_slack`, `integer_tolerance`,
`check_sat`, `direction_norm`, `subproblem_presolve`). An unknown key, a key the
chosen method does not read yet, and any `gdp_options` with a reformulation method
raise `ValueError` at entry. `loa` (#1775 G4.1, `discopt.gdp.algorithms.loa.LOA`) reads `iterlim` (default: `max_nodes`), `init_algorithm` (default `no_init`), `set_cover_iterlim` and discopt's own `subproblem`: `"global"` (the default) certifies each configuration's reduced subproblem with discopt and falls back to the local NLP step when that proves nothing; `"nlp"` is the pre-G4.1 loop (fixed-integer NLP on the big-M lowering). Measured on the `gdp_parity` panel at 60 s: `nlp` 8 solved, `global` 12, 0 incorrect. `loa`, `gloa`, `ric` and `ldsda` also read discopt's own `subproblem_time_limit` (#1984): each global subproblem's wall budget in seconds, default a tenth of the run within [1 s, 30 s] (#1153, LBB's node budget). Before #1984 LOA's global subproblem had no cap, and one nested `Model.solve` took 849 s of a 3600 s run. A subproblem stopped by the cap is uncertified, never excluded as resolved. LOA logs one INFO line per iteration to the `discopt.solvers.gdpopt_loa` logger (master bound, selected disjuncts, subproblem outcome and time, bound and incumbent in model units). `gloa` (#1775 G4.3, `discopt.gdp.algorithms.gloa.GLOA`) reads `iterlim`, `init_algorithm` and `set_cover_iterlim`; its subproblems are always global, so it reads no `subproblem` key. `ric` (#1775 G4.2) reads the same three keys; `enumerate` reads only discopt's own `max_configurations` (an integer ≥ 1). `lbb` (#1775 G4.4, `discopt.gdp.algorithms.lbb.LBB`) reads `iterlim` (a node limit) and `check_sat` (a bool; default on here, since logic propagation is exact and replaces Pyomo's z3 check). `ldsda` (#1775 G4.5, `discopt.gdp.algorithms.ldsda.LDSDA`) reads Pyomo's `starting_point`, `logical_constraint_list` and `disjunction_list` (the *names* of `exactly(1)` logical constraints and root disjunctions; Pyomo takes the components), `direction_norm` (`"L2"` or `"Linf"`), `iterlim`, `check_sat`, and `subproblem` (`"local"`, the default, or `"global"`). With no lists, every root exactly-one disjunction is an external variable; with no `starting_point`, each group takes its first value the logic does not refute.

**Axis 2 — `solver` / `mip_nlp_method`: *how the resulting algebraic MIP/MINLP is
solved.*** With `solver="mip-nlp"`, `mip_nlp_method` selects the decomposition
algorithm: `oa`, `ecp`, `fp`, `goa`, `lp_nlp_bb` (implemented); `roa` (reserved).

### The two locked rules

1. **Native-disjunctive `gdp_method` + `solver="mip-nlp"` raises.** Requesting a
   native `gdp_method` (any of `NATIVE_GDP_METHODS`, reserved ones included) together with `solver="mip-nlp"` is contradictory —
   one says "solve the disjunctions natively," the other says "reformulate to
   algebraic and decompose." This raises a clear `ValueError` rather than silently
   reformulating away the native request.
   Enforced: `solver.py` `_validate_gdp_method` at `solve_model` entry (before the
   reserved-name refusal, so the conflict is what a caller sees), and again in the
   mip-nlp route's `native_gdp_methods` guard.
   Tests: `test_mip_nlp_rejects_native_gdp_solver_method` (`python/tests/test_mip_nlp.py`),
   `test_rule_1_covers_every_native_method` (`python/tests/test_1775_g3_1_gdp_routing_and_base.py`).

2. **`goa` ≠ `gloa`; they must stay distinct and must not be aliased.** `goa` is a
   `mip_nlp_method` (generalized/global OA over the *algebraic* MINLP). `gloa`
   (global *logic-based* OA) belongs on the *native* `gdp_method` axis, where it
   runs since #1775 G4.3 — requesting it as a `mip_nlp_method` raises and points the caller at
   `goa` (algebraic) vs the `gdp_method` axis (logic-based).
   Enforced: `mip_nlp.py` `_normalize_method` (~line 638).
   Test: `test_mip_nlp_method_gloa_is_reserved_for_gdp_axis` (`python/tests/test_mip_nlp.py`).

Line anchors are approximate — grep the named guards/tests if they drift. The
umbrella "establish the two-axis API" work is tracked in
`mip-nlp-upstream-readiness.md` (issue #111).
