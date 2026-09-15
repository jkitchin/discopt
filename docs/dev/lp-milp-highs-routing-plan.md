# Route pure LP/MILP to HiGHS — plan (v2, 2026-09-15)

Status: **proposed**, nothing implemented. Branch `feat/lp-milp-highs-route` (from
`main` @ `a837ffb8`). Reverses the "pure-Rust LP/MILP" policy of #356/#365 and
`docs/design/pounce-only-roadmap.md` goal 4 for the **user-facing pure LP/MILP class
only**. The MINLP kernels (convex, spatial, per-node simplex, FBBT, OBBT, NS-safe
bounds) are out of scope and are not modified.

Targets #1183 (MILP search competitiveness) and #1229 (sentinel readback). Interacts
with #1212, #1230, #1231, #1141 (§11).

Every file:line below was read at `a837ffb8` on 2026-09-15. Wall-clock figures from
`milp-competitiveness-plan.md` §4 were taken at machine load 66–69 and are **not**
cited; node counts are load-independent and are the numbers that carry weight.

---

## 0. Why the policy changes

1. **The gap is search, not the LP.** #1183's first comment reverses its own
   diagnosis: with the cut pool disabled, HiGHS 1.14 needs 516 nodes from a root of
   71 where the Rust driver needs 2,667 from a root of 169.4; the Pyomo path takes
   4,973 nodes vs 2,667 engine-direct. On the MIPLIB-38 panel HiGHS solved 38/38 in
   15,417 nodes vs the Rust driver's best arm 21/38 in 4,693,240 nodes
   (`milp-competitiveness-plan.md` §4, A2–A14 mostly falsified). Closing a ~300×
   node gap means re-implementing HiGHS's presolve, cut loop, propagation,
   restarts and heuristics — against an MIT-licensed, pip-installable solver that
   already exists.
2. **Owner priority.** The product is the MINLP solver ("I don't want to change the
   MINLP code, it is pretty good"). Pure LP/MILP is the entry class users hit first
   and the class where we are least competitive.
3. **Certificate risk concentrates in the pure-MILP path.** #1229 (a free column
   returned at the `1e20` sentinel, labeled optimal) is in the Rust driver's LP
   readback; the driver's optimality bound is floating-point (§1.4). Routing these
   classes to a mature solver **plus a solver-independent verifier** (§3) lowers the
   false-certificate surface.

#365's three reasons for dropping HiGHS were packaging, differentiability, and
"verify a certificate, don't trust a solve". Speed was never among them. Answers:
packaging → §5 (cp312 wheels exist on all five release targets since highspy 1.7.2; 1.7.1 lacked linux aarch64); differentiability
→ `differentiable_solve` on a MILP fixes integers and differentiates the LP, which
needs duals — HiGHS supplies them (§3.1.7, parity-tested); certificates → §3, which
trusts HiGHS for nothing that enters the certificate except the MILP tree bound,
which is labeled.

Not a reason: "HiGHS is faster per LP". On small warm-started node LPs the in-house
simplex is competitive (`performance-plan.md` §16 puts the QPLIB_1157 loss in the
*factorization*, not the pivot loop). Per-node replacement is rejected in §2.2.

---

## 1. What is on the path today (verified by reading)

### 1.1 Dispatch

- `Model.solve` → `solve_model` (`python/discopt/solver.py:7377`; kwargs
  `time_limit=3600, gap_tolerance=1e-4, threads=1` (reserved), `deterministic=False,
  max_nodes=100_000, nlp_solver="pounce", lagrangian_bound, initial_point,
  solver=None`).
- `solver="gurobi"` dispatch at ~10078 → `_solve_lp_gurobi` (19623) /
  `_solve_milp_gurobi` (20407). **This is the precedent the HiGHS route copies.**
- Pure LP: `solver.py:10179` `return _solve_lp(model, t_start, time_limit,
  prefer_pounce=nlp_solver == "pounce")`. `_solve_lp` (19466) tries
  `[_solve_lp_simplex, _solve_lp_pounce]` (reversed when `prefer_pounce`), carries
  the `_DeferredUnbounded` logic of #850/#937, and ends in `status="error"`.
- Pure MILP: `solver.py:10188–10280`. `_want_engine = nlp_solver == "simplex" or
  (_milp_engine_default_on() and not lagrangian_bound)` (`_milp_engine_default_on`
  22198 re-reads `DISCOPT_MILP_ENGINE` per call). Callbacks force the spatial B&B
  (#748). Engine = `_solve_milp_simplex` (22371) → Rust `solve_milp_csc_py`;
  otherwise `_solve_milp_bb` (22859, Python B&B).
- **Reformulation routes re-enter the same dispatch**: `solver.py:9357`, `9544`,
  `9701` adopt `model = _bml` and set `nlp_solver = "simplex"`, so a nonconvex model
  that reformulates exactly to a MILP falls into the branch above. Any change at
  10188 reaches them unless gated (§2.2).
- `mip_nlp.py:287` calls `solver_module._solve_lp`; `:339` calls
  `solver_module._solve_milp_simplex`. The MINLP fast paths call the *engine
  functions*, not the dispatch, and are untouched by a dispatch-level change.
- `pyomo/solver.py:184` calls `discopt_model.solve(**solve_kwargs)`: Pyomo users
  get whatever the dispatch does.

### 1.2 Marshalling

- LP: `_solve_lp_matrix` (19654) builds **dense** `A_eq_full = _dense_A(lp_data.A_eq)`
  (19692) and `_decompose_eq_slack_form`s it before `solve_lp_fn(c, A_ub, b_ub, A_eq,
  b_eq, bounds, time_limit)`. All three LP engines (simplex 19571, pounce 19590,
  gurobi 19623) go through it. **The pure-LP path densifies today**: netlib `dfl001`
  (6,071 × 12,230) is a 594 MB dense array. The HiGHS route must not inherit this
  (§3.1.1).
- MILP: `_csc_for_rust_milp` (22336) hands standard-form CSC to the Rust driver; the
  HiGHS route reuses it (§3.2.1).

### 1.3 Verification that already exists

- `_solve_lp_matrix` gates OPTIMAL on `_matrix_solution_feasible` (20091; per-row
  `|viol| ≤ 1e-6 + 1e-9·scale`), logs "returned an infeasible point labeled optimal;
  falling back to the next engine" and returns `None`.
- `_solve_milp_simplex` gates on `_point_feasible` (22558; `_gate_tol = 1e-5`,
  integrality 1e-4) **using the sparse `lp_data.A_eq` when sparse** — the matvec
  #1229 shows reporting a 173.82 violation on a row whose true residual is 0 (FMA
  contraction on ±1e20 products). On failure: "deferring to a sound engine",
  returns `None` → Python B&B.
- `_solve_milp_gurobi` (20407) has **no point gate at all**.
- `Model.solve`'s final incumbent guard (`modeling/core.py:3288`) re-checks the point
  against the *original* model and on failure withholds `x`, forces
  `gap_certified=False`, `status="error"`. It is the last line and stays.

### 1.4 Bound provenance — the draft's claim was wrong

The draft said the Rust MILP driver "uses Neumaier–Shcherbina safe bounds". It does
not. `ns_safe_bound` / `ns_safe_bound_csc` (`lp/simplex/refine.rs:217/268`) are
called only from `bnb/convex_kernel.rs:38`, `bnb/spatial_kernel.rs:27`,
`bnb/spatial_tree.rs`. `milp_driver.rs:27` imports `solve_lp, solve_lp_cols,
solve_lp_cols_scaled, solve_lp_warm, solve_lp_warm_scaled_csc, tighten_bounds_csc`
and defines `farkas_safe_bound{,_csc}` (`:3320/3375`) for *infeasibility* pruning
only. Its optimality dual bound is floating-point, exactly as HiGHS's
`mip_dual_bound` is. **Routing MILP to HiGHS does not lower the provenance class of
the pure-MILP certificate.** It does not raise it either; §3.2.5 adds a cheap safe
root check and labels the result.

### 1.5 The existing HiGHS wrapper

`python/discopt/solvers/milp_highs.py` (676 lines; opt-in master engine via
`get_milp_solver(backend="highs")` in `lp_backend.py`). Facts that matter for reuse:

- Sets only `output_flag`, `mip_rel_gap`, `time_limit` (status-checked).
  `max_nodes` is accepted and **unused**. `h.setSolution(sol)` return is unchecked.
- `objective = float(c_arr @ x)` — **no objective offset**. Bound = `mip_dual_bound`
  clamped `≤ objective` on OPTIMAL.
- `_status_map`: `kUnboundedOrInfeasible → UNBOUNDED` (must be disambiguated,
  §3.1.9), `kSolutionLimit`/`kInterrupt → ITERATION_LIMIT`, `kObjectiveBound/Target
  → CUTOFF`.
- `_to_highs_inf`'s comment says "a literal 1e20 becomes finite"; HiGHS's own
  `infinite_bound` default is `1e20` (probe). Harmless; comment inaccurate.
- `solve_milp_with_lazy_cuts` is a restart loop on `kCallbackMipImprovingSolution`
  + `kCallbackMipInterrupt` (~3000 polls/s). `kCallbackMipDefineLazyConstraints` is
  declared (`ref/HiGHS` v1.14.0-4: `HConst.h:242`, `highs_bindings.cpp:1769`,
  `highspy/highs.py:1326`) but the MIP solver never fires it — MIP fires only
  `kCallbackMipSolution` (`HighsMipSolverData.cpp:834/1196`) and
  `kCallbackMipImprovingSolution` (`:2594`). Lazy cuts stay a restart loop.

A deleted pure-LP wrapper `solvers/lp_highs.py` (256 lines, `solve_lp(c, A_ub, b_ub,
A_eq, b_eq, bounds, warm_basis, time_limit) -> LPResult` with duals, reduced costs,
basis) exists at `1ebef93c` (removed in `5cb8b537`). A starting point, not a drop-in:
it speaks the dense `_solve_lp_matrix` interface.

### 1.6 highspy facts (probe `scratchpad/highspy_probe.py`, local 1.12.0)

Present: `getDualRay`, `getPrimalRay`, `getBasis`/`setBasis`, `getRanging`,
`setSolution`, `changeObjectiveOffset`, `readModel`, `getLp`, `presolve`/`postsolve`,
callbacks. `HighsLp.offset_` is applied to `objective_function_value`. Options
accepted: `mip_max_nodes`, `mip_abs_gap`, `mip_rel_gap`, `random_seed`, `threads`,
`mip_feasibility_tolerance` (1e-6), `primal_feasibility_tolerance` (1e-7),
`objective_bound`, `time_limit`, `presolve`, `solver`, `run_crossover`,
`infinite_bound` (1e20), `small_matrix_value` (1e-9), `large_matrix_value` (1e15).
`setOptionValue("no_such_option")` returns `HighsStatus.kError` (and prints) — every
set must be status-checked. `HighsInfo` exposes `max_primal_infeasibility`,
`max_integrality_violation`, `mip_dual_bound`, `mip_gap`, `mip_node_count`,
`simplex_iteration_count`, `primal_solution_status`, `basis_validity`. Infeasible LP:
`getDualRay() → (kOk, True, ray)`. `HighsModelStatus` includes `kSolveError`,
`kModelError`, `kPresolveError`, `kPostsolveError`, `kMemoryLimit`,
`kHighsInterrupt`, `kUnknown`.

PyPI (`scratchpad/highspy_pypi.json`): latest 1.15.1 (2026-07-02),
`requires_python >= 3.9`. **1.7.0 ships no wheels** — the draft's `highspy>=1.7`
floor is unusable for a core dependency. 1.7.1 (40 wheels + sdist) and 1.7.2 (50 +
sdist) already ship wheels and an sdist; 1.8.0 (50) and 1.8.1 (60) ship wheels but
**no sdist**; 1.9.0 (60 + sdist) is the first release after 1.7.x with both; 1.12.0+ linux tags are `manylinux_2_24/2_28`; 1.15.x ships 60 wheels,
≤ 7.2 MB each. #1183 measured 157 nodes on 1.12 vs 109 on 1.14 for its probe, so the
version is a benchmark variable.

### 1.7 Corpus facts

- MINLPLib has no pure LP/MILP; the in-repo corpus is **66** `.nl` (draft: 61).
- `ref/HiGHS/check/instances`: 101 files (78 `.mps`, 18 `.lp`), includes
  `issue-2388.lp` (#1229's instance; HiGHS: optimal 0.0, 0 nodes).
- The MIPLIB-38 harness `scratchpad/miplib/{loader.py,hs.py,screen.py,miplib_ab.py}`
  named in `milp-competitiveness-plan.md` §6 is **absent from disk**; §0 there records
  how it was built (MIPLIB 2017 easy, size-filtered, each instance validated by HiGHS
  reproducing the published optimum from the converted arrays, TL 20 s).
- `benchmarks.toml`: `[suites.lp_netlib]` (112) with gates `lp_netlib_pass_rate` and
  `lp_vs_highs_geomean ≤ 3.0` (191–192) against `[solvers.highs]
  command="/Applications/AMPL/highs"` (472) — an external binary, not highspy.

---

## 2. Scope and staging (design question 1)

### 2.1 Why HiGHS and not SCIP

| | HiGHS | SCIP |
|---|---|---|
| License | MIT | Apache-2.0 |
| pip | `highspy`, ≤ 7.2 MB, 60 wheels, no system deps | `pyscipopt` bundles the SCIP shared library with SoPlex/PaPILO/…, ~10× larger |
| API needed | one object: `passModel`/`run`/`getSolution`/`getDualRay` | constraint-handler model; LP duals via the transformed problem |
| Role here | already an opt-in master engine; the MIPLIB-38 oracle | the **external reference** for MINLP benchmarks (CLAUDE.md 3-way head-to-head) |

Making SCIP a core dependency would erase the independence of the benchmark
reference. HiGHS is the only defensible *core* dependency; SCIP stays the yardstick.

### 2.2 Stages

**Stage 0 — measure first (no solver change).** Rebuild the LP/MILP harness as a
checked-in script (§6.1), validate the loader, record the Rust baseline on all
panels, land the failing #1229 dense regression test, and run E0/E1 (§6.4). Worth
doing even if the branch is abandoned.

**Stage 1 — user-entry pure LP/MILP → HiGHS.** New engine functions
`_solve_lp_highs` / `_solve_milp_highs` beside the Gurobi pair, selected at the two
dispatch points (`10179`, `10188–10280`). Two flag flips:

- *1a*: models classified LP/MILP **at entry** (the user's model, not a `_bml`
  reformulation). Judged under the bound-changing bar (§7): cert-clean +
  net-positive on the LP/MILP panels; **exact bound-neutral on the MINLP guard**
  (`node_count`, certified `objective` unchanged on every MINLP instance), which
  holds because no MINLP instance reaches the entry LP/MILP branch.
- *1b*: the three reformulation routes (`9357/9544/9701`). These are MINLP-entry
  instances whose certificate changes engine; judged cert-clean + net-positive on
  the subset of the 66-`.nl` corpus that takes those routes (Stage 0 records which,
  via the routing counter of §6.1); exact bound-neutral on the complement.
  Optional; if the subset is empty or the panel is neutral it stays on the Rust
  driver and the measurement is recorded.

The MINLP fast paths in `mip_nlp.py` call the engine functions directly and are
**not** rerouted (owner constraint).

**Stage 2 — optional, flag-gated: HiGHS as the default MILP master for OA / AMP /
GDP-LOA.** The seam exists: `get_milp_solver(backend="auto")` (`lp_backend.py`).
Add `highs` to the `auto` order under `DISCOPT_MILP_MASTER_BACKEND={auto,highs,
simplex}` (default `auto` = today's order until graduation). Callers that pass
`auto`/`milp_solver`: `oa.py` (×3), `gdpopt_loa.py`, `_relax/milp_relaxation.py:737`,
`partition_selection.py`; `amp.py` reaches it via `MilpRelaxationModel`
(`_backend_kw`, `amp.py:1000/1841`) and its `_valid_milp_solvers` (`:2330`) must
gain `highs`. Lagrangian and Benders pin `backend="simplex"` and are untouched by an
`auto` change. Evidence for trying: #1141 "in-house master ~3× slower than HiGHS"
(`performance-plan.md` §25) and rsyn0840m root-gap closure 86.1 % (HiGHS master) vs
0 % (#1060). This is a *master* change, not a per-node change: the outer loop
already treats the master's point and bound as fallible and re-verifies. Entry
experiment E2 (§6.4) gates whether Stage 2 is built at all; graduation per
`flag-graduation-protocol.md`. Separate PR; separately abandonable.

**Stage 3 — rejected: HiGHS as the MINLP per-node LP.** Requires linking HiGHS into
the Rust kernels (C++ in the maturin wheel build; `release.yml` is pure `abi3` on
five targets), touches every kernel, and no measurement says the pivot loop is the
bottleneck (§0). Recorded so it is not re-proposed without new evidence.

### 2.3 What "done" means for the branch

Stage 0 landed; Stage 1a default-ON after graduation; highspy a core dependency
with per-platform wheel smoke; #1229 closed (dense test passes on the default
route; Rust readback refuses loudly so the opt-out route cannot reproduce it either);
#1183 re-scoped to "pure MILP routed to HiGHS; Rust driver search work closed unless
reopened with new evidence". Stages 1b and 2 are optional increments; the branch is
mergeable without them.

---

## 3. Soundness contract (design question 2) — binding

Principle (CLAUDE.md §1, §3): HiGHS's *labels* are never trusted for a certificate.
Every certified field of `SolveResult` is either (i) recomputed by discopt from the
returned point with a solver-independent kernel, or (ii) labeled with its provenance
so a consumer can refuse it.

### 3.1 LP (`_solve_lp_highs`)

1. **Marshal sparse.** Standard form from `LPData` (`c, A_eq, b_eq, x_l, x_u,
   obj_const`; `problem_classifier.py:164`) → CSC via `_csc_for_rust_milp` → HiGHS
   `passModel` colwise, `row_lower = row_upper = b_eq`, `offset_ = obj_const`.
   Sense is handled by discopt (negated `c`, as today) — never `changeObjectiveSense`,
   so the dual sign convention stays the internal-minimization one that
   `_lp_qp_unpack_duals` (19011) expects. `±1e20` bounds map to `kHighsInf`
   (`_to_highs_inf`, kept). No `_dense_A`, no `_decompose_eq_slack_form`.
2. **Options** (each `setOptionValue` return checked `== kOk`, else `RuntimeError`
   naming the option): `output_flag=False`, `threads` left at the HiGHS default (§12,
   global scheduler), `random_seed=<tuning
   seed>`, `time_limit = max(0.1, time_limit − (now − t_start))`,
   `primal_feasibility_tolerance=1e-7`, `dual_feasibility_tolerance=1e-7`,
   `run_crossover="on"` (a vertex is required for duals/basis), `presolve` and
   `solver` at defaults. `Highs().version()` goes to `solver_stats["highs/version"]`.
3. **Readback guard (the #1229 class).** Before any residual: `max|x| < 1e15` and
   `max|dual| < 1e15`, else `status="error"` naming the index ("readback at sentinel
   magnitude"). This makes the verifier's matvec choice immaterial — the FMA
   artifact in #1229 needs products ~1e20 to survive a 1e-6 tolerance.
4. **Feasibility verification** of `x` on the standard-form data after guard 3:
   `|A_eq x − b_eq|_i ≤ 1e-6 + 1e-9·scale_i` (the `_matrix_solution_feasible`
   convention), `x_l − 1e-6 ≤ x ≤ x_u + 1e-6`. The #1229 regression test and a
   unit test feeding a sentinel vector through the verifier pin that the guard fires
   first. Failure → `status="error"`, violation and row in the message, `x`
   withheld, **no fallback** (§4).
5. **Objective recomputed**: `objective = c·x + obj_const` (negated for MAXIMIZE) as
   `_solve_lp_matrix` does — never `objective_function_value`.
6. **Dual bound — NS-safe.** From HiGHS `row_dual`/`col_dual`, compute the
   Neumaier–Shcherbina safe lower bound through **one new Rust binding**
   `ns_safe_bound_csc_py` over `refine.rs:268`. This aligns with #1230 ("nothing that
   enters the certificate is computed in Python"; its five Python NS twins become
   consolidation targets). Certificate: `bound = NS(...) + obj_const`, `gap =
   objective − bound`, `gap_certified = gap ≤ 1e-6 + 1e-9·|objective|`. If the NS
   bound is further below the objective, the result is `status="feasible",
   gap_certified=False` with both numbers — an honest near-optimal, never a false
   optimal. `solver_stats["lp/ns_gap"]` records the gap. **Kill criterion**: if
   > 2 % of HiGHS-OPTIMAL L-Netlib instances decertify under NS at these
   tolerances, the tolerance is *not* loosened; the cause (scaling, crossover) is
   analysed and recorded in §12 before proceeding.
   *Refined after the kill fired (§12):* when the NS bound over the declared box is
   `-inf` or too loose, two further bounds from the same dual are tried. Each is a
   valid lower bound for any `y`, and the largest valid one is used.
   (a) **NS over an FBBT box.** Open sides are replaced by sides implied by `A x = b`
   and the declared box (`fbbt_box`). Each derived side is widened by a bound on its
   whole floating-point error, so the box contains the feasible set exactly.
   (b) **Exact rational dual correction** (`exact_ns_bound`, project-and-shift in the
   style of Steffy–Wolter). Wrong-signed open columns get reduced cost exactly zero from
   a rational solve on pivot rows chosen in floating point. The bound is then summed in
   rationals and rounded down.
   The tolerance is unchanged. A bound from (a) or (b) sets
   `algorithm_route` label `lp/bound_provenance` and a numeric stat. Caps:
   ≤ 256 columns, ≤ 8 rounds, and a time budget; past a cap the result stays
   `feasible`.
7. **Duals.** `dual_values = row_dual`, `reduced_costs = col_dual` (probe: signs
   match on the 1-row LP; the parity test in §7 asserts equality with the Rust
   simplex duals within 1e-6 on nondegenerate netlib instances and equal objective
   on degenerate ones). Unpacked via `_lp_qp_unpack_duals`. `convex_fast_path=True`.
8. **Infeasible.** `kInfeasible` accepted only with a verified Farkas certificate:
   `getDualRay()` → `(kOk, True, r)`, then discopt checks `rᵀA` against the bounds
   and `rᵀb` produce a contradiction with margin ≥ 1e-9·scale. Verified →
   `status="infeasible"`, `gap_certified=True` (the one non-gap certified status,
   `core.py:3033`), `infeasibility_certificate=<ray>`. Not verified →
   `status="error"` ("infeasible label without a verifiable ray").
   *Refined (§12):* the ray is also checked over the FBBT box of §3.1.6(a). If no ray
   verifies after the presolve-off re-solve, a **phase-1 proof** is tried:
   `min 1ᵀ(p+q)  s.t.  A x + p − q = b` over the FBBT box of `A x = b` has value 0 when
   the LP has a point. An NS or exact-corrected bound above `1e-9·(1 + ‖b‖₁)` therefore
   proves the LP empty. Provenance `lp/infeasible_provenance` ∈
   {`farkas-fbbt-box`, `phase1-ns-fbbt-box`, `phase1-exact-dual-correction`}. Anything
   else is still `error`.
9. **Unbounded.** `kUnbounded` → `getPrimalRay()`, verified `A d = 0`, `c·d < 0`,
   bound directions respected → `status="unbounded"`. `kUnboundedOrInfeasible` →
   one re-solve with `presolve="off"`; still ambiguous → `status="error"`. The
   `_DeferredUnbounded` box-relaxation logic (#850/#937) is engine-agnostic and stays.
10. **Limits.** `kTimeLimit`/`kIterationLimit` with a point that passes 3–4 →
    `status="time_limit"` (the string `_solve_milp_gurobi` already uses),
    `gap_certified=False`; without → `status="time_limit"`, no `x`. `kSolveError`,
    `kModelError`, `kPresolveError`, `kPostsolveError`, `kMemoryLimit`, `kUnknown`,
    `kHighsInterrupt` → `status="error"` carrying the HiGHS status string.

### 3.2 MILP (`_solve_milp_highs`)

1. **Marshal** as `_solve_milp_simplex` does up to the Rust call (term classifier +
   `model_to_repr` linearity backstop, `_csc_for_rust_milp`, `int_idx`, `n_orig`,
   `obj_const`), then `passModel` with integrality. `initial_point` → `setSolution`,
   **return checked**; a rejected start is a warning, not an error.
2. **Options** (status-checked): `output_flag=False`, `threads` at default (§12),
   `random_seed`,
   `mip_rel_gap = gap_tolerance`, `mip_abs_gap = 1e-6`, `mip_max_nodes = max_nodes`
   (the existing wrapper drops it — fixed), `time_limit = remaining budget`,
   `mip_feasibility_tolerance=1e-6`, `primal_feasibility_tolerance=1e-7`.
3. **Readback guard + verification** of the incumbent as §3.1.3–4, plus
   integrality `|x_j − round(x_j)| ≤ 1e-5` on `int_idx` (the conftest tolerance;
   the Rust route's gate uses 1e-4 and the sparse matvec — both tightened on the new
   route only). Failure → `status="error"`, no fallback.
4. **Objective recomputed** from the point with `obj_const` and sense. HiGHS's
   `objective_function_value` is compared; a mismatch > 1e-6·(1+|obj|) is a warning
   with both values in `solver_stats`.
5. **Dual bound.** `bound = mip_dual_bound` (+`obj_const`, sense-mapped), clamped
   `≤ objective` only after 3–4 pass. Labeled
   `solver_stats["milp/bound_provenance"] = "highs-fp"` — the same floating-point
   class as the Rust driver's bound today (§1.4). `gap_certified = (kOptimal and
   verification passed)`, matching `_solve_milp_simplex`'s `gap_certified=status ==
   "optimal"`. *Cheap safe-root check, default ON*: solve the root LP relaxation with
   `_solve_lp_highs` (NS-safe) and assert `ns_root ≤ bound + 1e-6·(1+|bound|)`; a
   violation is `status="error"` ("tree bound below safe root bound"). This does
   not make the tree bound safe; it catches a root bound that is already wrong at
   one LP's cost, and supplies `root_bound` for parity (the Rust route does a second
   integer-relaxed `solve_milp_csc_py` for this today). Kill: if the Tiny panel
   shows the root LP > 20 % of median wall on sub-0.1 s instances, it moves behind
   `DISCOPT_HIGHS_ROOT_CHECK=0` with the measurement recorded, default still ON
   unless the owner overrules.
6. **Duals** on the MILP: `_mip_recover_relaxation_duals` (19165) as today, fed by
   `_solve_lp_highs` on the integer-fixed LP.
7. **Status map** (replaces `milp_highs._status_map` for this route). `kOptimal` →
   `"optimal"` after 3–5. `kInfeasible` → `"infeasible"`: a MILP has no Farkas ray;
   accepted as HiGHS's claim, labeled `solver_stats["milp/infeasible_provenance"]
   ="highs"`, `gap_certified=True` — the same standing the Rust driver's own claim
   has today (its Farkas prune is per node; presolve/propagation in both solvers is
   unverified). Owner decision §10.5 can make this stricter for both routes.
   `kUnbounded`/`kUnboundedOrInfeasible` → LP relaxation via §3.1.9.
   `kTimeLimit`/`kNodeLimit`/`kIterationLimit`/`kSolutionLimit`/`kInterrupt` with
   a verified incumbent → `status="feasible"`, `gap_certified=False`, `bound` kept
   (the contract at `core.py:1712`); without an incumbent → `"time_limit"` /
   `"node_limit"` with `bound`. `kObjectiveBound`/`kObjectiveTarget` are never set,
   so reaching them is `"error"`. All error/unknown statuses → `"error"`.
8. **Parity fields**: `node_count = mip_node_count`; `solver_stats = {"lp/iters":
   simplex_iteration_count, "lp/driver_nodes": mip_node_count, "highs/version",
   "milp/bound_provenance", "route/lp_milp_backend": "highs"}`; `root_bound` from 5;
   `wall_time`; `convex_fast_path=True`; `solution_pool` only if the restart loop
   is used (it is not, in Stage 1).

### 3.3 Reformulated routes (Stage 1b)

Same functions; only the gate in §2.2 differs. The mapping of the `_bml` result back
to the user's model at `9357/9544/9701` is engine-agnostic.

### 3.4 Things HiGHS is never asked to decide

Whether a point is feasible; the objective value; LP infeasibility or unboundedness
without a verified ray; whether the problem is an LP/MILP at all
(`classify_problem` + the linearity backstop stay upstream).

---

## 4. Fallback policy (design question 3)

CLAUDE.md §3 forbids silent approximation and swallowed failure. The tree has two
*logged* fallbacks on verification failure (`_solve_lp_matrix` → next engine;
`_solve_milp_simplex` → Python B&B); they exist because the Rust engines were the
thing being distrusted. On the HiGHS route:

- **No automatic HiGHS → Rust fallback.** Verification failure, an unverifiable ray,
  or a HiGHS error status returns `status="error"` with the reason. A fallback would
  hide the class of failure this plan exists to surface and would re-enter the
  engine #1229 is filed against.
- **Explicit opt-out kept**: `DISCOPT_LP_MILP_BACKEND=rust` selects the legacy path
  unchanged; `=highs` forces HiGHS; unset = the graduated default. Removed only with
  §8's deletion, never silently.
- **Kwarg**: `Model.solve(solver="highs")` mirrors `solver="gurobi"` and forces the
  route regardless of the env flag. `solver="rust"` is *not* added — two spellings
  of one switch is a dead flag waiting to happen.
- **Import failure**: highspy is a core dependency, so a failed `import highspy` is a
  broken install → `ImportError` with the install hint, raised at dispatch, never
  caught. The import stays lazy: a default MINLP solve must not import highspy
  (extend the existing "no jax on the default path" measurement to `highspy`).
- The two legacy fallbacks remain on the Rust route only and go with it in §8.

---

## 5. Packaging (design question 5)

- `pyproject.toml`: move `highspy` from `[project.optional-dependencies].highs` to
  core. **Floor `highspy>=1.10`** (was `>=1.9`; falsified, §12: 1.9.0 has no
  `getDualRay`/`getPrimalRay`, which the §3.1.8–9 certificates need). The sdist-less
  1.8.x line stays excluded. Keep the `highs` extra as an empty
  alias for one release (as `pounce` is).
- `release.yml`: after each wheel builds (ubuntu x86_64/aarch64 `manylinux: auto`,
  macos-14 x86_64/aarch64, windows x64, py3.12 abi3), install it in a fresh venv and
  solve a 3-variable MILP asserting `status=="optimal"` and
  `solver_stats["highs/version"]`. A wheel whose highspy dependency does not
  resolve fails the release, not the user.
- `ci.yml`: highspy is installed in one job today (`:297`); with a core dependency
  it is installed everywhere. Add one job running the suite with
  `DISCOPT_LP_MILP_BACKEND=rust` until §8 deletes that route; a version-matrix job
  runs the floor and the latest highspy.
- manylinux: highspy 1.12+ wheels are `manylinux_2_24/2_28`, discopt's are `auto`.
  Users on older glibc fall to highspy's sdist (CMake + C++). Documented, not solved.

---

## 6. Benchmark protocol (design question 4) — before and after

### 6.1 Harness `discopt_benchmarks/scripts/lp_milp_panel.py` (new, checked in)

Rebuild the lost `scratchpad/miplib/` harness as a committed script. Loader:
`.mps`/`.lp` via `highspy.readModel` → `getLp()` → arrays → `Model` via
`add_linear_constraints` (`modeling/core.py:4847`) with integrality and `offset_`.
**Loader validation**: for every instance, raw HiGHS on the *converted* model must
reproduce the oracle optimum within 1e-6·(1+|obj|); failures are excluded and listed,
never silently dropped. The script prints per-instance progress unbuffered, ends with
the executed-comparison count and exits non-zero when it is 0 (CLAUDE.md §6, §10).

Arms per instance, on the same converted model:
**R** Rust route (`DISCOPT_LP_MILP_BACKEND=rust`); **H** HiGHS route (`=highs`);
**H0** raw highspy on the arrays — the ceiling; H − H0 is marshalling + verification
overhead, reported separately.

Metrics: status, objective, bound, `gap_certified`, `node_count`, wall (median of 3
interleaved rounds, sd), solved-within-TL, SGM(wall, shift 1 s), median over solved,
total wall over all, per-instance `max_viol` from the script's *own* dense
verification, `incorrect_count` against the oracle (**0**).

Routing counter `solver_stats["route/lp_milp_backend"]` is set at the dispatch point
and asserted per instance. Marker check: the script asserts `discopt.solver.__file__`
and presence/absence of `_solve_milp_highs` for the arm under test (CLAUDE.md §8).

### 6.2 Panels

| Panel | Source | Purpose |
|---|---|---|
| M-HiGHS | `ref/HiGHS/check/instances` MILPs (`n_int > 0`, ≤ 20k nnz; ≈ 40) | search competitiveness, TL 20 s |
| M-MIPLIB | MIPLIB 2017 easy, size-filtered as `milp-competitiveness-plan.md` §0 (38 targeted; rebuilt, validated) | #1183 headline, TL 20 s |
| L-Netlib | `[suites.lp_netlib]` (112) | LP correctness, duals parity, NS decertification rate |
| Probe-1183 | the #1183 instance | node-count watch (H vs H0 must match up to restarts) |
| Probe-1229 | `issue-2388.lp` | the dense regression test |
| Tiny | 200 generated LP/MILPs, 2–20 variables, fixed seed | per-call overhead (object, marshal, verify, root check) |
| MINLP-guard | the 66-`.nl` corpus, certifying panel | **exact** bound-neutral on every instance not taking a reformulation route; cert-clean on those that do (1b) |

### 6.3 Discipline

`uptime` 1-min load **< 2** before every timed round (today 4–5: timing rounds wait);
abort if it passes 4 mid-round; record load in the header. Interleave R/H/H0 per
instance, 3 rounds, median + sd. Print `highspy.__version__`, `Highs().version()`,
commit. Never `/dev/null` stderr; growing log; `pgrep` before declaring a run dead.
The MINLP-guard runs the flag both ways and diffs the JSON; drift on a
non-reformulated instance is a bug in the gate, not a finding.

### 6.4 Entry experiments (before Stage 1 code)

- **E0**: Probe-1183 + 10 M-HiGHS instances, R vs H0 node counts (load-free).
  Expectation from #1183: H0 ≤ R/5 on Probe-1183. **Kill**: H0 not ≥ 3× fewer
  nodes on ≥ 7 of 10 → the headline motivation is not reproduced here; stop and
  re-scope.
- **E1** (Tiny): raw highspy per-call overhead on 2–20-variable problems vs the Rust
  driver. Expectation < 5 ms. **Kill**: H > 3× R on median Tiny wall *and* > 20 ms
  → Stage 1 gains a measured size gate (Rust below an nnz threshold), recorded as a
  §3 exception.
- **E2** (Stage 2 entry): on the 66-`.nl` instances whose OA/AMP/GDP run reaches a
  master solve, `backend="simplex"` vs `"highs"` through `get_milp_solver` only:
  outer iterations, wall, final certificate. **Kill**: not cert-clean, or < 30 % of
  instances improve wall ≥ 20 % with any regressing > 20 % → Stage 2 not built.

### 6.5 Pre-registered hypotheses

- **H1** (M-MIPLIB): H ≥ 35/38 in 20 s where R ≤ 25; SGM(wall) H ≤ R/3. Kill: H ≤
  R + 3 solved.
- **H2** (L-Netlib): objectives equal within 1e-6·(1+|obj|) wherever both solve; NS
  decertification ≤ 2 %; duals parity per §3.1.7.
- **H3** (Tiny): median H/R ≤ 3 and ≤ 20 ms absolute.
- **H4** (MINLP-guard): zero drift on non-reformulated instances; cert-clean on
  reformulated ones.
- **H5** (H vs H0): overhead ≤ 10 % of H0 wall above 1 s, ≤ 20 ms below.

Any failure is written to §12 before the next PR (CLAUDE.md §4, §11).

---

## 7. PR breakdown, tests, docs (design question 7)

**Owner decision (2026-09-15): one branch, one PR.** All rows below land as commits
on `feat/lp-milp-highs-route` and ship together in a single PR, including defects
found and fixed along the way (e.g. #1229, P0b). The rows remain the unit of work and
of verification; they are no longer separate PRs. The PR description reports every
row's tests and the §6 panel results.

Each row: `pytest -m smoke`, the adversarial suite, `cargo test -p discopt-core` when
Rust is touched; regression tests fail-before/pass-after.

| PR | Scope | Tests |
|---|---|---|
| **P0** `bench: LP/MILP panel harness + loader validation + baseline` | `lp_milp_panel.py`, corpus manifest, baseline JSON under `discopt_benchmarks/results/`, E0/E1 results in §12 | harness self-test on 3 instances: comparison count > 0, loader validation fires |
| **P0b** `test(correctness): #1229 dense regression` | `issue-2388.lp` at the **driver entry** (`_marshal_std_form` → `solve_milp_csc_py`, the path `get_milp_solver("simplex")` callers take) *and* through `Model.solve`; dense verify ≤ 1e-6; `max|x| < 1e15`; `xfail(strict=True)` on the driver entry only | **Done on this branch (§12, 2026-09-15):** root cause fixed in `dual_slack_basis`; the driver-entry tests fail before and pass after, so no `xfail`. The readback guard was withdrawn — it trips on a second, separate sentinel defect in the shared dual engine (`select_leaving`), which is MINLP-path code and gets its own issue |
| **P1** `feat(lp): _solve_lp_highs (default-off)` | `solvers/lp_highs.py` revived, sparse, §3.1; `ns_safe_bound_csc_py` binding; env flag + `solver="highs"`; routing counter | option-status error; sentinel-vector refusal; Farkas / primal-ray verification; offset + sense; duals parity on 5 netlib; P0b `xfail` passes under `=highs`; default MINLP solve imports no `highspy` |
| **P2** `feat(milp): _solve_milp_highs (default-off)` | §3.2: `mip_max_nodes`, status map, root NS check, `setSolution` check | each status branch on hand-built instances; node/time limit → `feasible`/`gap_certified=False`; integrality gate; Tiny overhead budget |
| **P3** `feat(packaging): highspy core dep + wheel smoke` | `pyproject.toml`, `release.yml`, `ci.yml` jobs | CI |
| **P4** `flag: graduate DISCOPT_LP_MILP_BACKEND=highs (1a)` | flip only, per `flag-graduation-protocol.md`; panel JSON attached | `graduation-gate.yml` — **Done on this branch (§12)** |
| **P5** (opt.) `flag: 1b reformulated routes` | gate removal at 9357/9544/9701 | MINLP-guard both ways |
| **P6** (opt.) `feat(masters): highs in get_milp_solver auto (flag)` | Stage 2; `amp.py:2330` allowlist | E2 + graduation |
| **P7** `chore: delete dead LP bindings` | ~~§8.1 only~~ **Re-scoped (§12, 2026-09-15):** the §8.1 bindings are test entries to kept engine code; no deletion. Only the import probes switch to the CSC bindings their wrappers call. **Done (§12)** | smoke; import-probe tests |
| **P8** `docs` | CLAUDE.md, `solve-routing.md`, `pounce-only-roadmap.md` goal 4, `milp-competitiveness-plan.md` closure note, §12 here | **Done on this branch.** The CLAUDE.md Architecture edit uses the text drafted below and is flagged for owner review in the PR |

Doc edits needing owner sign-off (P8): the CLAUDE.md Architecture note ("default
per-node LP engine is the in-house Rust simplex … do not plan work against a 'HiGHS
backend'") stays true for MINLP and gains: "Pure LP/MILP models classified at entry
are routed to HiGHS (`docs/dev/solve-routing.md`)". `pounce-only-roadmap.md` goal 4
retired with a pointer here; `solvers/` description updated.

---

## 8. Deletion policy (design question 6)

### 8.1 Removable after P4 (or dead already at `a837ffb8`)

> **Falsified 2026-09-15 (§12).** "Zero package references" was true of
> `python/discopt/` alone. The tests and benchmark harnesses use most of these bindings
> as the test surface of the kept `lp/simplex` engine (§8.3), so deleting them removes
> coverage of code that stays. P7 is re-scoped in §12. The original text is kept below.

`crates/discopt-python/src/lp_bindings.rs`: `crossover_to_vertex_py` (132),
`recover_basis_py` (167), `solve_lp_py` (376), `profile_counters_py` (731),
`profile_reset_py` (742) — zero package references. `solve_milp_py` (1110) — only
import probes (`lp_simplex.py:36`, `lp_backend.py:152`), which switch to
`solve_milp_csc_py`. `solve_lp_warm_py` (504) — docstring only.

### 8.2 Removable only when the Rust MILP route is retired (owner decision)

`_solve_milp_simplex`, `_solve_milp_bb`, `_SIMPLEX_MILP_BUDGET_CAP_S`,
`_milp_engine_default_on`/`DISCOPT_MILP_ENGINE`, `DISCOPT_MILP_SWAP_RESEED`, the
second integer-relaxed root solve, the cut bindings `gomory_cuts_py`/`mir_cuts_py`/
`aggregation_mir_cuts_py` (209/269/323), `solve_lp_batch_py` (751; used by
`lp_simplex.solve_lp_batch` 324–360 — audit MINLP users first), the `parallel`
feature (`milp_driver.rs`, `lp/simplex/batch.rs`, `examples/par_bench.rs`,
default-on in both `Cargo.toml`). `_csc_for_rust_milp` stays (HiGHS uses it).
**Not** `milp_driver.rs`'s entries `solve_milp_csc_py`/`solve_milp_lazy_csc_py`/
`run_milp_hooked` (1239/1399/1531) while `get_milp_solver("simplex")` users exist —
Lagrangian, Benders, OA/AMP masters all do today. Retiring the driver needs Stage 2
graduated *and* the Lagrangian/Benders pins moved, which is MINLP code the owner
said not to touch. **Recommendation: 8.1 on this branch; 8.2 not.**

### 8.3 Keep

`lp/simplex/*` (per-node engine), `refine.rs` NS bounds (now also serving the HiGHS
LP route), `farkas_safe_bound*`, `tighten_bounds*` (`driver/presolve.rs`, `mod.rs`),
`obj_integral`, FBBT/OBBT, `spatial_*`, `convex_kernel`.

### 8.4 Impact on #1212 / #1229 / #1230

- #1212 (standalone `.nl → MILP` binary): `crates/discopt-core/src/bin/nl_milp.rs` is
  **uncommitted and untracked** at `a837ffb8`; it hands to `solve_milp_csc`. Unaffected
  by Stage 1; affected only by 8.2, in which case #1212's Phase 1 target no longer
  exists and the issue is closed or re-pointed at HiGHS's CLI (§10.4).
- #1229: the defect is on the driver entry (`_marshal_std_form` → `solve_milp_csc_py`)
  that MINLP masters, Lagrangian and Benders use; `Model.solve` does not hit it on
  `issue-2388` (§12). Only the Rust readback PR closes it; P0b + P1 add guards.
- #1230: the Python NS twins (`milp_simplex.py:365/137`, `obbt.py:60`) become
  consolidation targets onto `ns_safe_bound_csc_py`; not done here, but the binding
  is designed for it.

---

## 9. Risks

| Risk | Mitigation / kill |
|---|---|
| HiGHS point fails our verifier on a valid instance (its 1e-7 primal tolerance vs our 1e-6+1e-9·scale; postsolve drift) | rate measured on L-Netlib/M-MIPLIB in P1/P2 and reported; > 0.5 % blocks P4 and is investigated; tolerances are not loosened |
| NS bound decertifies HiGHS-optimal LPs (crossover, scaling) | §3.1.6 kill at 2 %; crossover enforced; results labeled `feasible`, never `optimal` |
| Tiny-model overhead slows the interactive case | E1/H3 kill → measured size gate, recorded exception |
| highspy version drift changes nodes and, rarely, statuses | version in every report; CI matrix floor + latest; `solver_stats["highs/version"]` |
| Wheel/manylinux mismatch strands a platform | P3 per-platform smoke; floor 1.10 keeps the sdist escape hatch |
| Reformulated-route flip alters MINLP certificates | Stage 1b separate; cert-clean bar; complement exact-neutral |
| Deletion removes what #1212 / Lagrangian / Benders need | 8.2 deferred; 8.1 only |
| Harness silently measures nothing again | executed-comparison counter; routing counter per instance; marker check; non-zero exit on 0 |
| Load on this machine invalidates timing (4–5 today) | load gate < 2; node counts are the primary competitiveness metric |

---

## 10. Open decisions for the owner

**Resolved by the owner (2026-09-15):**

1. highspy floor: **`>=1.9`**, raised to **`>=1.10`** after 1.9.0 was measured to lack
   the ray getters (§12).
2. Stage 1b: **build, behind a flag** (P5), graduation per §2.2.
3. Stage 2: **run E2 first**; P6 is built only if E2 passes its kill criterion.
4. Deletion depth: **§8.1 only**; §8.2 is not done on this branch.
5. MILP `kInfeasible`: **certified with a provenance label** (§3.2.7).
6. Root NS check: **default ON**, revisited only if E1 shows it dominates Tiny wall.

The original questions, kept for the record:

1. **highspy floor**: `>=1.9` (recommended: sdist + wheels) or `>=1.14` (matches
   #1183's measurement; newer manylinux tag).
2. **Stage 1b** (reformulated routes to HiGHS): build, or leave on the Rust driver?
3. **Stage 2** (HiGHS default master for OA/AMP/GDP): run E2, or not at all?
4. **Deletion depth**: 8.1 only (recommended) vs 8.2 (retires the Rust MILP driver as
   a Python entry; moves Lagrangian/Benders pins; re-points #1212).
5. **`kInfeasible` on a MILP**: accept HiGHS's claim with `gap_certified=True` and a
   provenance label (§3.2.7, matches the Rust driver's standing today), or make MILP
   infeasibility `gap_certified=False` from *any* engine (stricter; changes the Rust
   route too).
6. **Root NS check default** (§3.2.5): ON (recommended) unless E1 shows it dominates
   tiny-model wall.

---

## 11. Issue interactions

- #1183: re-scoped by §2.3. Its Rust-search items close as superseded once P4 lands;
  its Pyomo-path node inflation (4,973 vs 2,667) is re-checked on the HiGHS route —
  if it persists it is a marshalling difference, not an engine one.
- #1229: root cause fixed on this branch (`dual_slack_basis`, §12); the branch's
  single PR closes it. P1 adds the same regression guard on the HiGHS route.
- #1230: the LP/QP boundary contract (`docs/dev/lp-qp-boundary.md` on unmerged
  `claude/bold-lovelace-9f1wzw`) gains one Rust binding to consolidate onto.
- #1231: unaffected (MINLP twin loops).
- #1212: see §8.4.
- #1141 / #1060 (closed): their HiGHS-master evidence is the basis for Stage 2.
- #356 / #365 (closed): the pure-Rust policy is superseded for the entry LP/MILP
  class only; the MINLP per-node policy stands.

## 12. Falsification log

**2026-09-15 — #1229 does not reproduce through `Model.solve`; P0b re-targeted.**
Commit `a837ffb8`, highspy 1.15.1, venv built from this branch (`discopt.__file__`
asserted). Loader (`discopt_benchmarks/scripts/lp_milp_loader.py`, MPS/LP → text
`.nl` → `from_nl`) validated by round-trip (`Model.to_mps` → HiGHS) on 8 instances
(2 LP, 6 MILP): 8 compared, 0 mismatches. On `issue-2388.lp`:

- `Model.solve` (default route; `_solve_milp_simplex` called once, returned
  `optimal`, 1 node): obj 0, max|x| 10.2, dense max violation 3.5e-14, 0 columns at
  the sentinel. **Feasible.**
- The issue's direct driver call (`_marshal_std_form` → `solve_milp_csc_py`, same
  arguments): `optimal`, obj 0, 8 columns at 1e20, dense max violation 10.2.
  **Reproduces.**

The draft's claim that the P0b test "fails today" through `Model.solve` is
retracted: the user entry marshals through `_csc_for_rust_milp`, not
`_marshal_std_form`, and avoids the defect on this instance. The defect is live on
the driver entry that `get_milp_solver("simplex")` callers use, so Stage 1 does
**not** close #1229 by itself; the Rust readback PR (P0b row) is required.

- The public `discopt.solvers.milp_simplex.solve_milp` (the MINLP master /
  Lagrangian / Benders entry) on the same arrays: `SolveStatus.OPTIMAL`, obj 0,
  bound 0, 1 node, 8 columns at 1e20, dense max violation 10.2. **Reproduces** —
  a false `optimal` is reachable from MINLP code today, independent of this plan.

**2026-09-15 — #1229 root cause found and fixed at the source (folded into this
branch by the owner).** The readback was not the defect: `dual_slack_basis`
(`milp_driver.rs`) labeled a zero-cost column with `l = -INF` as `AT_UPPER` even
when `u = INF` too, so every readback (`dual.rs` `assemble`/`reduced_rhs`) returned
`u_j = 1e20`. A free column now takes `AT_LOWER`, the engine's "free, sits at 0"
encoding (`primal.rs` `run`/`nb_value`).
Regression: `python/tests/test_issue_1229_free_column_sentinel.py` (arrays frozen in
`data/issue1229_issue2388_milp.npz`) — 2 failed / 1 passed on the pre-fix build
(`max|x| = 1e20`), 3 passed after; Rust unit test
`dual_slack_basis_never_parks_a_free_column_at_the_sentinel` fails with the one-line
fix reverted (`col_status` 2 = `AT_UPPER`, expected 0) and passes with it. The P0b "separate
Rust PR" is superseded by this fix; the `xfail(strict=True)` plan is moot.

**2026-09-15 — the "refuse loudly at readback" guard was tried and withdrawn; it
exposed a second sentinel defect in the shared dual engine.** A guard in
`dual.rs::assemble` (downgrade `Optimal` → `Numerical` when a nonbasic column reads
back at `|x| ≥ INF`) failed 4 `dual.rs` unit tests that pass at `a837ffb8`
(`unstable_pivot_recovery_is_not_gated_on_a_deadline`,
`dual_stall_bail_can_cost_a_bound_when_the_cold_solve_fails`,
`cost_perturbation_breaks_the_degeneracy_on_the_captured_stall`,
`cost_perturbation_is_verdict_neutral_on_every_captured_lp`), all on the captured
`qplib2170_cold_fail_lp.json`. Probe: the start basis is correct (351 free zero-cost
columns, all `AT_LOWER`); during the warm loop basic columns with `[0, 1e20]` bounds
drift past the sentinel (x_B up to ~1e24) and `select_leaving` (`xb > u + tol`)
treats `u = 1e20` as a real bound, pinning them `AT_UPPER`. The solve ends `Optimal`,
obj 0 (HiGHS: 0), with 16 columns at `x = 1e20` — the tests pass only because they
check status and objective. Out of scope here: that engine is the MINLP per-node LP,
which the owner froze for this work. Needs its own issue and fix (`select_leaving`
must not treat `u ≥ INF` as a bound; readback guard afterwards).

**2026-09-15 — E0 (Stage-0 agent; 20 s, gap 1e-4, 1 round, load 6.4 → 19.5).**
22 runs, **1 incorrect, on the Rust route**: `2122.lp` (max-sense, 855 columns, 257
integers, |a| in [8e-3, 2.9e4]) is certified `infeasible` at 0 nodes, reproducibly;
HiGHS optimum −187616.11, SCIP on the same `.nl` −187612.94, so the model file is
right and the false certificate is the Rust MILP route's (build with the #1229 fix).
Node ratio R/H0 on the 10 most node-heavy M-HiGHS instances: ≥3× on 6/10 (bell5
≥2522×, dcmulti 4297×, gesa2 ≥76001×, sp150x300d ≥35571×, lseu 1648×, flugpl 5.0×).
Below 3×: issue-2173 (2.4×) and issue-2095 (1×). The other two are R *failures*
where a ratio is undefined: issue-2446 (no incumbent after 20 s at the root; HiGHS
0.06 s) and 2122 (false infeasible). Probe-1183: 4965 vs 90 nodes (55×). The §6 E0
criterion read literally is not met (6/10); counting an R failure as an H0 win it
is (8/10). **Restated here, before any decision: an arm that fails (a false
certificate, or no incumbent where the other arm certifies) loses that instance.**
Under that reading E0 passes 8/10.

**2026-09-15 — E1 (provisional; load 15.7/15.0, not a valid timing gate).** 200
tiny generated instances, 3 interleaved rounds: `Model.solve` (R) median 4.97 ms
(sd 3.78), raw HiGHS arrays (H0) 0.38 ms (sd 5.24); all 200 objectives agree. This
compares a whole pipeline against a bare engine and says nothing yet about arm H's
overhead; rerun on arm H under the load gate.

**2026-09-15 — P1/P2 implementation decisions that refine §3 (recorded before the
PR).**

- `solver_stats` stays `dict[str, float]` (its type and every benchmark consumer):
  provenance is numeric flags (`route/lp_milp_backend=1.0`,
  `milp/bound_provenance_highs_fp=1.0`, `milp/infeasible_provenance_{highs,farkas}`,
  `highs/version` as `MMmmpp`); the human-readable labels go to
  `SolveResult.algorithm_route`.
- Readback guard (§3.1.2) refined by an existing pin: `min -x` over the default box
  is `optimal` at `−9.999e19` (`test_lp_huge_finite_box_937`, `test_850…`). A value
  with |x_j| ≥ 1e15 is accepted only when it *equals* a declared finite bound
  (|bound| < 1e20); anything else at that magnitude, and any dual ≥ 1e15, is
  `error`.
- The LP route solves the model's own standard form (A x = b with logicals) rather
  than the ub/eq projection, so the NS certificate is about exactly what HiGHS
  solved; duals are mapped to the projection's layout through
  `_slack_row_orientation`, now the single statement of the projection's row rule
  (bound-neutral refactor: `test_crossover.py` + `test_lp_std_form_consolidation.py`
  28 passed).
- HiGHS runs with `random_seed=0` (SolverTuning has no seed). `threads` was pinned to
  1 here and is now left at the default; see the §12 global-scheduler entry.
- An LP `kInfeasible`/`kUnbounded` without a verifiable ray is re-solved once with
  presolve off (rays exist only on the unreduced model) before becoming `error`.
- `kSolutionLimit` is what HiGHS returns for `mip_max_nodes` (probe, 1.15.1) and
  maps with `kIterationLimit`/`kInterrupt` to `node_limit`; `kTimeLimit` to
  `time_limit`; either is `feasible` with a verified incumbent. `mip_max_nodes` is
  clamped to 2³¹−1 (HiGHS returns kError above it).
- The HiGHS MILP route is skipped for an explicit `nlp_solver="simplex"` (names the
  Rust engine) and for `lagrangian_bound` (needs the per-node path), and callbacks
  still fall through (#748). Selection is `DISCOPT_LP_MILP_BACKEND={rust,highs}`,
  default `rust` until the graduation panel; an unknown value raises.

**2026-09-15 — falsified: the float feasibility convention is not a certificate at
the default box (found by the route's own tests, before any panel).**

- Measured: `min x − y s.t. x + y ≥ 2, x + y ≤ 1` over free columns (default
  `±9.999e19` box) returned `optimal` −1.9998e20, `gap_certified=True`,
  `lp/ns_gap = 0.0` on the HiGHS route. The Rust route returns `infeasible`.
- Mechanism: HiGHS's point is the corner `x = −9.999e19, y = 9.999e19`. The row
  activity cancels to exactly 0 in floating point, and the
  `_matrix_solution_feasible` tolerance `1e-9·Σ|A||x| = 2e11` forgives the violation
  of 2. The NS bound certifies optimality *given* feasibility, so it cannot catch this.
- This retracts the readback bullet above as a sufficient guard: a huge value "on a
  declared bound" can still carry a cancelled violation. That bullet is also
  superseded for logicals (below).
- Fix, in `lp_milp_highs`:
  - (a) A row touching |x| ≥ 1e15 is residual-checked in exact rational
    arithmetic, with the huge values left out of its tolerance scale. In that row the
    logical (continuous, zero cost, single row) is re-derived exactly and checked
    against its bounds, because its double cannot hold e.g. `9.999e19 + 1`.
  - (b) The readback guard exempts logicals, whose value is never trusted.
  - (c) Bound checks apply only to declared sides: a `2e20` logical is not above the
    `1e20` "no bound".
  - (d) An LP optimum whose point fails, on a model with huge declared bounds, is
    re-solved once with those bounds relaxed to infinity and presolve off. Only a
    Farkas ray verified against the *declared* bounds is used (→ `infeasible`);
    anything else is `error`.
  - (e) An unverifiable MILP incumbent consults the root LP the same way
    (→ `infeasible`, provenance `farkas-root-lp`).
- Residual limit, not claimed sound: rows with every |x_j| < 1e15 keep the float
  convention, the same forgiveness `_matrix_solution_feasible` gives every other
  engine.
- Regression tests: `test_rows_at_the_huge_box_are_checked_exactly`,
  `test_lp_infeasible_needs_a_verified_farkas_ray`,
  `test_milp_on_the_huge_box_gets_a_farkas_proof_not_a_bare_label`,
  `test_milp_incumbent_that_fails_verification_is_an_error`,
  `test_lp_default_box_corner_is_optimal_not_error`, and
  `test_lp_huge_finite_box_937.py` run with the flag set to `highs`.
- Verified after the fix:
  - The route file passes 19/19.
  - `test_lp_huge_finite_box_937.py`, `test_850_cross_backend_unbounded_and_feasibility.py`,
    `test_issue_1229_free_column_sentinel.py` and `test_solver_duals.py` pass 46/46 under
    both `DISCOPT_LP_MILP_BACKEND=rust` and `=highs`. Before the fix, #937's default-box
    test failed on `highs`.
  - `pytest -m smoke` on the default backend: 1444 passed, 12 skipped, the same as before
    the branch.
- Mutation check: `feasibility_problem` replaced by the pre-fix float-only version, with a
  load marker asserted. Detections: **3 of 3** (the two LP/kernel tests and the MILP
  huge-box test).
- Retraction: on the first mutation run the MILP huge-box test (then named
  `…_never_certified_from_a_cancelled_point`) passed under the mutant, so it did not meet
  the fails-before bar.
  - Probed on four variants: HiGHS's MILP presolve returns `kInfeasible` by itself, so
    the status is `infeasible` either way.
  - What the fix changes is the provenance: `farkas-root-lp` with the fix, a bare `highs`
    label without it. The test now pins that provenance.
  - The incumbent-failure branch (e) is not reachable from any instance found. It is
    tested directly by forcing the incumbent check to refuse, which gives `error` with
    `milp/root_check_ran = 1` on a feasible root LP.

**2026-09-15 — contract gaps: `pytest -m smoke` with `DISCOPT_LP_MILP_BACKEND=highs`.**

- Setup: highspy 1.15.1, load 7.9 on 14 cores.
- Result: 1443 passed, 12 skipped, 1 failed. The adversarial suite passed 19/19 on the
  default backend.
- The one failure is
  `test_mo_augmecon2.py::test_lexicographic_differs_from_simple_under_alternative_optima`.
  It is not a soundness gap; the test pinned which f1 optimum the LP engine returns:
  - The Rust route returned x2 = 2.29989832, an interior point of the optimal face.
  - HiGHS returned the vertex x2 = 2.
  - Both are f1-optimal, so on HiGHS the simple payoff happened to coincide with the
    lexicographic one.
- Fix, in the test: the simple payoff is fed the documented alternative optimum x2 = 3
  explicitly. The AUGMECON2 distinction is now tested without depending on the solver.
  It passes 6/6 on both backends.

**2026-09-15 — the #1229 driver fix on MINLP: sound on the whole corpus; hda's
time-limit bound gets looser, for a reason in the MINLP layer.**

- Setup: a TEMP build that logs every `dual_slack_basis` call and the free columns it
  relabels. `DISCOPT_1229_OLD=1` restores the pre-fix label in the same binary.
  - Positive control on the #1229 fixture: the old arm reproduces `|x| = 1e20`; the
    fixed arm gives 10.2.
  - Corpus: `python/tests/data/minlplib_nl`, 66 instances, TL 60 s.
- Result:
  - 26 instances reach `dual_slack_basis`. Only **hda** has a free column the fix
    relabels.
  - hda is the only instance whose outcome differs. Interleaved A/B at TL 60 s, load 7.1 →
    6.7:
    - old arm: bound −64509.8 in 4 of 4 runs
    - fixed arm: −1.399e10 in 3 of 4, −122962 in 1 of 4
  - No oracle flags in either arm; every bound is below the hda optimum.
  - Under `deterministic=True` the time limit does not cap the solve (the role-2 clocks
    are off), so that A/B produced no output and was killed.
- Mechanism (probe wrapping `solve_milp_csc_py`, `MilpRelaxationModel.solve` and
  `solve_at_node`):
  - The call is the root node's pure LP (0 integers, 3658 rows) on the cold route of
    `MilpRelaxationModel.solve`.
  - **Old label:** the Rust driver returns `node_limit`, with no point and no bound. The
    node layer then re-solves on the warm route (most likely the C-42 retry after dropping
    the pool rows). That solve reaches `optimal` with an NS safe bound of −64509.8.
  - **Fixed label:** the same LP solves `optimal` at −64509.85 (row and bound
    violations ≤ 2e-7). No retry runs.
    - `milp.solve` attaches the stale safe bound from the failed warm attempt,
      −7.2e10 (#362 `_pending_numerical_bound`).
    - After square-tangent separation it is −2.48e16.
    - `_certify` accepts it: valid but loose.
- Conclusion:
  - The fix is correct: the old label made a solvable LP fail.
  - The looser hda bound comes from the MINLP layer. When a cold `optimal` has only a
    stale safe bound, it counts as certified, which prevents the retry that would find a
    tight one.
  - Per the owner's instruction, the MINLP code is left unchanged here. This is reported
    as a follow-up.
- Retraction: probes v1–v3 showed a bound of −141697.4 in both arms, and I read that as
  load sensitivity. That was wrong: the probe altered the solve.
  - Its recorder raised after the real call returned (it assumed the returned `x` had
    `ncols` entries; it has the 1338 structural entries).
  - The exception was swallowed by `except Exception: return res` in
    `_separate_convex` (`mccormick_lp.py:2639`); `_separate_univariate_square` (2270) has
    the same swallowing handler.
  - The node therefore took a different path. The CLAUDE.md §3 finding is reported, not
    fixed here, since it is MINLP code.
  - Probes v4 onward never raise into the solver, and they reproduce the uninstrumented
    A/B bounds.

**2026-09-15 — falsified: the `highspy>=1.9` floor.**

- Measured: the route and certificate files in a venv pinned to highspy 1.9.0 gave
  1 failed / 45 passed (`test_lp_huge_finite_box_937::test_simplex_unbounded_verdict_is_never_deferred`).
- Cause: 1.9.0 has no `getDualRay` / `getPrimalRay`, and the §3.1.8–9 certificates
  need both.
- highspy 1.10.0: 65 passed, and 46 passed with `DISCOPT_LP_MILP_BACKEND=highs`.
- The floor is now `>=1.10` (pyproject, §5, §10). The owner's `>=1.9` decision is
  superseded by this measurement.

**2026-09-15 — L panel, round 1, statuses only.**

- Setup: 24 LPs, arms R/H/H0, TL 20 s, commit `a837ffb8`, highspy 1.15.1. Load
  7.1 → 6.1, so this run is not valid for timing.
- Soundness: `incorrect_count = 0` (72 runs).
- **The §3.1.6 kill fired.** H returned `feasible` (NS not closing) on
  **11/24 = 46 %**, where HiGHS itself said optimal: 25fv47, adlittle, afiro, e226,
  etamacro, israel, scrs8, stair, standata, standgub, standmps. The tolerance was
  not loosened.
- R: `time_limit` on 25fv47, shell and standmps. Its reported objectives are above H0
  by up to 4.4e-5 (scrs8: 904.29699798 vs 904.29695380). That is within the panel's
  rel-1e-4 oracle, so it is recorded here, not counted as incorrect.
- **Cause**, probed on the four hardest (25fv47, e226, scrs8, stair):
  - Each has **zero-cost recession directions**. Every exact optimal dual has
    reduced cost exactly 0 on some open-sided columns.
  - A float `y` puts roundoff-size reduced costs on the open side, so the NS term is
    `-inf`. No choice of float dual avoids this.
  - Evidence: a cost perturbation of 1e-9 on those columns makes all four
    `kUnbounded`.
- Entry experiments (each exits non-zero on zero comparisons or any bound above the
  H0 optimum):

  | Method | Certified | Unsound |
  |---|---|---|
  | NS over an FBBT box | 6–7 / 11 | 0 |
  | FBBT box + lsqr dual repair | 7 / 11 | 0 |
  | Basis repair | 0 / 4 | 0 |
  | Cost-perturbed re-solve | 0 / 4 | 0 |
  | **Exact rational dual correction over the FBBT box** | **11 / 11** | **0** |

  Exact correction took at most 0.05 s. It needed [1, 0] wrong-signed columns per
  round on 25fv47, [45, 11, 2, 0] on scrs8 and [17, 5, 1, 0] on stair.
- **Fix** (§3.1.6 refined): `fbbt_box` and `exact_ns_bound` in `lp_milp_highs`.
  - The implemented `fbbt_box` does not use the entry experiment's relative 1e-9
    widening. It widens by a bound on the full floating-point error of each derived
    side.
  - Reason: at `|a·x| ~ 1e19` the cancellation in a row total is off by far more than
    `1e-9·|side|`.
  - The numbers below were re-measured on the implementation.
- **Route probe** (`scratchpad/route_cert_probe.py`, `solve_lp_std`, TL 60 s):
  - All 11 are `optimal`, `gap_certified`, and no bound is above the H0 optimum.
  - Provenance: `ns-fbbt-box` on 7, `ns-exact-dual-correction` on 4 (25fv47, e226,
    scrs8, stair).
  - Whole-solve wall ≤ 0.14 s each.

**2026-09-15 — infeasible/unbounded panel (18 instances × R/H, load 6.1): 36
compared, 0 wrong; H returned `error` on 6 infeasible LPs.**

- The 6: bgetam, cplex1, forest6, klein1, refinery, vol1. Error is the honest refusal
  (no verifiable Farkas ray, even after the presolve-off re-solve), not a wrong
  answer. It is still a certification regression against R, which says `infeasible`.
- Fix (§3.1.8 refined):
  - (a) the ray is also verified over the FBBT box;
  - (b) otherwise, a phase-1 LP bound above `1e-9·(1+‖b‖₁)` over the FBBT box of
    `A x = b` is the proof.
  - The phase-1 bound can exceed HiGHS's phase-1 objective (bgetam: 486356 vs 54.3).
    Any bound above zero is a proof, so this is expected.
- Route probe: 6/6 `infeasible`, `gap_certified`.
  - `farkas-fbbt-box` on 5.
  - `phase1-exact-dual-correction` on klein1: its float NS is `-inf` even over the
    box; the wrong-signed columns per round were [22, 11, 8, 5, 0].
- Regression tests:
  - `test_exact_dual_correction_recovers_a_bound_float_ns_cannot`,
    `test_exact_dual_correction_is_a_valid_bound_from_any_dual`,
    `test_fbbt_box_contains_the_feasible_set_under_huge_term_cancellation`;
  - `test_zero_cost_recession_lp_certifies_by_exact_dual_correction` (e226);
  - `test_infeasible_lp_without_a_declared_box_farkas_ray_is_proved` (klein1, forest6).
  - The standard forms are frozen in `data/lp_highs_certificate_instances.npz`.
  - Fails before, passes after (`scratchpad/fails_before.sh`, which swaps in the
    pre-change module and asserts the `def exact_ns_bound` marker is absent):
    6 failed before (the fixture tests return `feasible` / `error`, and the kernel
    tests have no function to call); 6 passed after.
- Verified on the final code:
  - route + #1229 files: 28 passed;
  - `pytest -m smoke`: 1444 passed, 12 skipped, 0 failed, under both
    `DISCOPT_LP_MILP_BACKEND=highs` and `=rust`;
  - adversarial suite: 19 passed (load 6.3).

**2026-09-15 — M panel, round 1** (24 MILPs, TL 20 s, gap 1e-4, load 3.3 → 7.7, not
timing-valid).

- H: 24/24 `optimal`, 0 incorrect.
- R:
  - 1 incorrect, 2122 false `infeasible` (the E0 finding, still present with the
    #1229 fix);
  - `feasible` at the limit on bell5, gesa2 and sp150x300d;
  - `time_limit` with no incumbent on issue-2446.
- Nodes, R / H: i1183 4965 / 19, lseu 8241 / 15, p0548 11309 / 1, dcmulti 21485 / 1.

**2026-09-15 — H5 (bypass built): discopt's root presolve dominates the HiGHS route on
some MILPs.**

- issue-2446: H wall 6.26 s vs H0 0.065 s.
- The cProfile attributes 5.0 s to `PyModelRepr.presolve` in `run_root_presolve`,
  which runs before the pure-MILP dispatch.
- It was profiled under load, so the cause was measured again before any bypass was
  built.
- **A/B, confirmed.** `scratchpad/presolve_ab.py`, `Model.solve(presolve=True|False)`,
  flag `highs`, TL 20 s, a fresh subprocess per arm, 3 interleaved rounds, load gate
  passed (start 1.72, end 2.06). Kill: presolve off not ≥ 2× faster.

  | Instance | presolve on (median, sd) | presolve off (median, sd) | ratio |
  |---|---|---|---|
  | issue-2446 | 5.603 s, 0.000 | 0.581 s, 0.003 | 9.6× |
  | 2122 | 2.510 s, 0.018 | 0.675 s, 0.002 | 3.7× |

  Status, objective, bound and nodes are identical across the two arms on both
  instances (issue-2446: −785.3552363096325, bound equal, 1 node). issue-2446's on-arm
  wall equals the presolve cap `min(max(0.25·TL, 2), 30)` = 5 s plus the solve.
- **Bypass.** `_highs_takes_pure_lp_milp` (solver.py) repeats the HiGHS dispatch
  conditions before root presolve:
  - flag `highs`, and no `solver="gurobi"`;
  - class LP; or class MILP with no `nlp_bb=True`, no lazy / incumbent callback, no
    `nlp_solver="simplex"`, no `lagrangian_bound`, and `_milp_is_exactly_linear`.

  When they hold, it sets `presolve = False`, which also skips coefficient tightening
  and reverse-AD. Nothing after presolve reads `_presolve_stats`, and the presolved repr
  goes only to NLP-BB, which is excluded. A model that leaves the route after all keeps
  its declared box, which is valid but looser. The flag `rust` path is unchanged.
- The remaining off-arm wall (issue-2446: 0.58 s vs H0 0.065 s) is discopt overhead
  outside presolve. It is measured by H5 in the rerun panels.
- Regression test `test_highs_route_skips_discopt_root_presolve[lp,milp]`: before, 2
  failed (HiGHS arm called `run_root_presolve`, 2 == 1); after, 2 passed.
- Verified with the bypass in place:
  - route + issue-1229 tests: 30 passed;
  - `pytest -m smoke`: 1444 passed, 12 skipped, on both `rust` and `highs`;
  - adversarial suite: 19 passed;
  - infeasible panel: 36 compared, 0 wrong. Line-for-line against the pre-bypass run,
    the only changes are the 6 H `error` → `infeasible` fixed above (bgetam, cplex1,
    forest6, klein1, refinery, vol1). The 4 loader parse errors (issue-2874-3, 1448.lp,
    gams10am, garbage.lp) are the same on both arms and in both runs.

**2026-09-15 — §8.1 "dead bindings" (falsified): they are the tests' entry to the kept
engine.** `grep -rnw` over `python crates docs discopt_benchmarks .github`:

| Binding | Non-package users |
|---|---|
| `solve_lp_py` | `test_adversarial_recent_fixes.py:309`, `test_1017_farkas_cancellation_margin.py`, `test_simplex_lp.py` |
| `solve_lp_warm_py` | `test_simplex_lp.py` (warm-start and bad-basis tests), `e0_export_lp.py` |
| `crossover_to_vertex_py`, `recover_basis_py` | `test_rust_crossover.py` |
| `solve_milp_py` | `test_milp_simplex.py`, `test_node_propagation_default.py`, `test_928_…`, `bench_engine.py`, `perf/milp_node_efficiency.py`, 2 benchmark tests |
| `profile_counters_py`, `profile_reset_py` | none; documented permanent instrumentation (`issue-956-followthrough-plan.md:487`) |

The engine behind them (`lp/simplex/*`, the MILP driver) stays per §8.3. Deleting its
bindings deletes correctness tests of kept code, which CLAUDE.md §1 forbids.
**P7 re-scoped:** no binding deletions on this branch. The two package import probes
(`lp_simplex.py:36`, `lp_backend.py:152`) may still switch to `solve_milp_csc_py`,
because the default path no longer uses the dense entry. Any deletion waits for §8.2,
the retirement of the Rust MILP route, which is the owner's call.

**2026-09-15 — retraction: 2122 H-arm bound "above the oracle" was not a false
certificate.** While reading the M panel I flagged 2122's H bound as lying above both
its incumbent and the manifest oracle. 2122 is a `maximize` model, so bound ≥ incumbent
is the valid direction. HiGHS at `mip_rel_gap=0` gives −187612.944, which is exactly
what the H route returned. The manifest oracle −187616.11 is an incumbent at gap 1e-4
(§6.2), not the optimum. The sense-aware `_incorrect` judge counted it correctly; my
reading was wrong.

**2026-09-15 — `threads=1` broke the route in any process that ran HiGHS first (fixed).**

- Found on an in-process 2122 reproduction: the route returned `error` with HiGHS status
  `kNotset`. HiGHS log: "Option 'threads' is set to 1 but global scheduler has already
  been initialized to use 7 threads".
- Cause (`ref/HiGHS/highs/lp_data/Highs.cpp:852`, `:1046`): HiGHS keeps one
  process-wide scheduler, sized by the first `run()`. A later run whose nonzero
  `threads` differs is refused. `threads=0` (the default) is accepted. The OA/GDP
  paths, user code and the panel harness all run highspy with other settings, so a
  pinned `threads=1` made every later route solve in the process an `error`. The
  answer was never wrong, but the solve was lost.
- Fix: `_new_highs` leaves `threads` at the default. The panel's H0 and the loader's
  oracle were changed the same way. They had the same pin, and the tiny panel crashed
  on it.
- Regression test `test_highs_route_runs_after_another_highs_user_in_the_process[lp,milp]`
  (sizes the scheduler with a `threads=2` run, then solves through the route): before,
  2 failed (`'error' == 'optimal'`); after, 2 passed.
- Bound-neutral on the M and L panels (`scratchpad/cmp_threads.py`, H arm, 3 rounds,
  TL 20 s, highspy 1.15.1, load at end 3.7 / 2.7): 48 instances compared on status,
  objective, bound, `gap_certified`, route, nodes and per-round nodes, with 0 drift.
  144 + 144 executed, `incorrect_count` 0. H medians are unchanged within noise, e.g.
  issue-2446 0.599 → 0.603 s, 25fv47 0.837 → 0.818 s.

**2026-09-15 — P7 done as re-scoped; `solver="highs"` kwarg (§4) not built.**

- The import probes now name the binding each wrapper actually calls:
  `lp_simplex.py` probes `solve_lp_warm_csc_py` (what `solve_lp` imports) and
  `lp_backend._milp_simplex` probes `solve_milp_csc_py` (what `milp_simplex.solve_milp`
  calls). Before, both probed the dense `solve_milp_py`. Checked:
  `SIMPLEX_AVAILABLE True`, `_milp_simplex()` resolves.
- §4's `Model.solve(solver="highs")` is not added. With the default graduated (P4),
  it would be a second spelling of the default. That is the "two spellings of one
  switch" §4 itself rejects for `solver="rust"`. The escape hatch stays the env flag.

**2026-09-15 — P4: `DISCOPT_LP_MILP_BACKEND` graduated to default `highs`.** Base
`a837ffb8`, highspy 1.15.1.

- *Cert-clean:* the L and M panels (3 rounds, TL 20 s, after the threads fix) have H-arm
  `incorrect_count` 0, 48/48 `optimal`, and 48/48 `gap_certified`. The infeasible panel
  compared 36 with 0 wrong.
- *Net-positive:* the R arm on the same panels has a false `infeasible` on 2122 in all
  3 rounds, plus `time_limit` or uncertified `feasible` on 25fv47, shell, standmps, bell5,
  gesa2, sp150x300d and issue-2446. H certifies every one of them.
- *MINLP guard (H4):* `discopt_benchmarks/scripts/lp_milp_minlp_guard.py` ran all 66 `.nl` files in
  `python/tests/data/minlplib_nl/` as R (`rust`), H (`highs`) and R2 (`rust`, noise control),
  one subprocess each. It compared status, objective, bound, nodes, `gap_certified` and
  route. Result: compared=66, drift=0, noisy=3. clay0303hfsg, tspn10 and tspn12 differ
  between R and R2 under load 9–14, and on each one H equals R or R2 or lies between them.
  Route is `None` on all 66: no MINLP reaches the HiGHS route.
- Default-route test `test_default_route_imports_highspy_only_for_pure_lp_milp`: the milp
  case failed before the flip (route `None`) and passes after. The minlp case passes both
  ways and asserts that highspy is never imported.
- CI: the highspy floor/latest lane runs the cross-backend soundness files on the `rust`
  opt-out, so the legacy route keeps its coverage. `wheel_smoke.py` clears the env var and
  requires the default route.
- **E1 (tiny-model overhead) not measured.** The load gate (< 3.0) never opened while a
  concurrent calphad job ran (load 9–16), and the run was stopped. No H5/E1 overhead claim
  is made. Rerun `lp_milp_panel.py tiny` on a quiet machine.
- Stage 1b and Stage 2 (E2) were not built. Both are optional under §2.3.

**2026-09-15 — audit of the performance evidence against §6.5: H5 fails; H1–H3 were
not run as registered.** Recomputed from the raw panel JSONs
(`scratchpad/before_after.py`). Setup: R/H/H0 interleaved, 3 rounds, TL 20 s, gap 1e-4,
base `a837ffb8`, highspy 1.15.1.

- **Loads broke the §6.3 gate** (< 2 to start, abort above 4): M 2.7 → 8.0, L 2.8 → 6.2.
  Walls below are not gate-valid. Status, nodes and objectives do not depend on load.

  | Panel | Certified optimal R / H | Total wall R / H | Geo-mean wall R/H | SGM (1 s shift) R / H |
  |---|---|---|---|---|
  | M (23 M-HiGHS + Probe-1183) | 19/24 / 24/24 | 85.0 s / 5.3 s | 5.68 | 1.258 s / 0.200 s |
  | L (24 HiGHS-check LPs) | 21/24 / 24/24 | 117.0 s / 3.2 s | 11.89 | 1.680 s / 0.123 s |

- **Objectives agree.** In 88 comparisons of R or H against H0, the worst relative
  difference is 1.7e-5, inside gap 1e-4. Where both R and H are optimal, no pair differs
  by more than 1e-6.
- **H1** is registered on M-MIPLIB (38) and was **not run**. On the 24-MILP proxy panel,
  H solves 5 more than R (the kill line is ≤ 3 more), and the geo-mean ratio 5.7 meets
  "H ≤ R/3". The shifted SGM ratio, 1.9, does not. The registration names no shift.
- **H2** is registered on L-Netlib (112) and was **not run**. On the 24-LP proxy panel,
  objectives are equal and NS decertification is 0/24. Duals parity was not measured on
  a panel.
- **H3 is not tested.** The gated tiny run `panel_tiny_h5.json` (load 2.47 → 2.52,
  200 instances, 200 objectives agree) has only arms R (median 4.11 ms) and H0
  (0.29 ms), with no H arm. **Retraction:** the P4 entry above and PR #1258 said "E1 not
  measured". E1's R/H0 comparison was measured; the H arm is what is missing.
- **H5 fails.** All 48 instances have H0 wall < 1 s, so the 20 ms bound applies to each.
  - H − H0 exceeds 20 ms on 20/24 MILPs and 17/24 LPs; the medians are 44 ms and 38 ms.
  - Worst: 25fv47 760 ms, issue-2446 542 ms, 2122 536 ms, issue-2173 451 ms.
  - The load cannot explain an excess 10–40× the threshold. Attributed below.
- The panel JSONs live in the session scratchpad, not the repo.

**2026-09-15 — H5 attributed: the excess is the shared `Model.solve` pipeline and HiGHS
runs, not certification.** `scratchpad/h5_wall.py` wraps the named stages with wall-clock
timers (cProfile was discarded: it charged HiGHS's threaded C time to `_set_options`).
One run each, load 4.4–4.5, so shares only.

| Instance | H − H0 | NBT on declared box | Convexity classify | Std-form extract | HiGHS `run` (calls) | All certificate checks |
|---|---|---|---|---|---|---|
| 25fv47 (LP) | 735 ms | 216 ms | 151 ms | 84 ms | 90 ms (1) | 44 ms (34 ms `exact_ns_bound`) |
| 2122 | 534 ms | 67 ms | — | 44 ms | 450 ms (3) vs H0 128 ms | 4 ms |
| issue-2446 | 555 ms | 152 ms | — | 194 ms | 54 ms (3) | 8 ms |
| lseu | 39 ms | 4 ms | — | 2 ms | 146 ms (3) | < 1 ms |

- The certificate checks are ≤ 44 ms on every instance, and < 10 ms on the MILPs.
- The largest shares are stages that every `Model.solve` pays before dispatch, on the R route too:
  - nonlinear bound tightening of the declared box (`_declared_box_tightening`), run here on a
    purely linear model;
  - convexity classification;
  - DAG → std-form extraction.
- On each MILP the route makes three HiGHS runs: one `solve_milp_std` and two `solve_lp_std`
  (the stage counts). On 2122 those runs take 3.5× H0's single run.
- **Verdict: H5 fails as registered.** A 20 ms bound is not reachable through `Model.solve` on
  these sizes without bypassing the pre-dispatch pipeline. That is a separate change (skip the
  NBT and classification for route-bound models); it is not made here. No overhead claim is made.

**2026-09-15 — the Rust route's 2122 false `infeasible`, fixed.** Rust presolve `fbbt_row`
(`lp/simplex/presolve.rs`) declared a box empty when `lo > hi + tol`, with `tol` the pivot
tolerance.

- 2122's rounded row data leave a 3.2e-9 crossing (row 307, col 51). The driver returned a
  certified `infeasible` for a MILP whose optimum is −187612.94.
- Fix: emptiness needs a crossing > `FEAS_TOL`. A smaller crossing is widened to the pair of
  endpoints (the #907 idiom); it is never replaced by a midpoint.
- Tests:
  - `crossing_within_feasibility_tolerance_is_not_an_empty_box` failed before and passes after.
  - `crossing_beyond_feasibility_tolerance_is_still_infeasible` pins the refusal.
- On the rebuilt binary, 2122 on `DISCOPT_LP_MILP_BACKEND=rust` now ends at `time_limit` with
  bound −187609.32. For this maximize problem that is sound (≥ the optimum).

**2026-09-15 — the shared dual engine's sentinel defect, fixed (`dual.rs` `select_leaving`).**
The leaving-row test compared `x_B` with the raw bound arrays, so a ±1e20 sentinel side counted
as a bound.

- Effect: a free basic column that drifted past 1e20 was pivoted out *onto* the sentinel.
- On the captured qplib2170 relaxation, before the fix:
  - recovery off → `Numerical`, with 172 columns at |x| ≥ 1e19;
  - recovery on → `optimal 0` after 17 794 pivots, with 44 columns parked at 1e20.
- After the fix, both arms are `optimal 0` in 613 pivots, max |x| 2.
- The defect was the only trigger of three mechanisms on that fixture: the unstable-pivot
  bail/recovery, the "degeneracy", and the stall bail. The tests built on it were re-targeted
  without weakening:
  - `free_basic_column_is_never_pivoted_onto_the_sentinel` (qplib2170): no column at the
    sentinel, 0 bails, 0 recoveries, < 2000 pivots. It fails under the old rule
    ("HiGHS certifies optimal 0 left: Numerical").
  - `unstable_pivot_recovery_is_not_gated_on_a_deadline`: a new fixture,
    `testdata/bchoco06_unstable_pivot_lp.json`, is a real bchoco06 node LP (1002×1323) with its
    warm basis, captured from a default MINLP solve that still reaches the mechanism after the
    fix. Recovery off gives (bails, recoveries) = (1, 0); on gives (0, 1); status and objective
    are the same. It passes under both rules.
  - `cost_perturbation_breaks_the_degeneracy_on_the_captured_stall` moves to the tspn12
    fixture: 1573 of 1727 pivots are degenerate unperturbed, 64 of 906 perturbed, with the same
    optimum.
  - `dual_stall_bail_can_cost_a_bound_when_the_cold_solve_fails` forces patience 256.
- **Retraction:** `COST_PERTURB_EPS`'s qplib2170 evidence was this defect. Unperturbed it now
  converges in 613 pivots, and perturbing costs 912. The doc comment says so; tspn12 and
  st_testgr3 remain the evidence.
- **MINLP guard, cert-clean** (`scratchpad/sentinel_ab_guard.py`): all 66 in-repo `.nl` files,
  arms OLD (both old rules) / NEW / NEW2, `backend=rust`, TL 20 s, max_nodes 300, oracle
  `minlplib.solu`.
  - Result: ran 66, oracle-checked arms 156, oracle violations 0, certification regressions 0.
  - Counter profiles differ on 14 instances, so the toggle fired.
  - Four rows changed:
    - beuster and tspn10 are noisy: NEW2 differs from NEW.
    - casctanks: 17 → 9 nodes, with the same bound 6.24966 and incumbent 9.16348.
    - nvs05: 73 → 63 nodes, same bound 2.70808; the incumbent improves from 1107.89 to 8.116
      (NEW = NEW2).
  - Both are `feasible` at the time limit under load ~11, so their node counts are time-bound.
- The recovery and stall mechanisms stay live on the corpus after the fix: recoveries on
  bchoco06 and tspn05; stall bails on 4stufen, bchoco06/07 and tspn05/08.

**2026-09-15 — hda "floor-as-fallback": falsified, not built.** The hypothesis: a node whose
cold solve is optimal keeps the #362 stale NS floor as `safe_bound`, and `_certify` would
certify `r.objective` if the floor were only a fallback.

- `scratchpad/hda_certify_chain.py` evaluated the rest of the `_certify` chain on every
  stale-floor optimal result (2 evaluated). Both have obj −64509.85, with safe_bound −7.2e10 and
  −2.5e16.
- Both **decline on magnitude**: `_max_finite_magnitude` is 2.1e11, above the 1e7 limit. No
  column is unbounded and nonlinear.
- The kill criterion is met: floor-as-fallback cannot tighten hda. hda stays as documented.

**2026-09-15 — #1183 evidence on the issue's own reproducer.** `scratchpad/i1183_evidence.py`
runs the issue's Pyomo model (`gdp.bigm`, `SolverFactory("discopt")`) in a fresh subprocess per
run, with arms interleaved over 3 rounds.

- Before (`=rust`): 4973 nodes in all 3 rounds, the issue's figure exactly; `optimal` 224,
  certified; 2.673 ± 0.060 s.
- After (default): 7 nodes in all 3 rounds; `optimal` 224, certified, on the verified HiGHS route;
  0.463 ± 0.010 s.
- Load was 6.3 → 6.8, above the gate, so the walls are indicative only.
- The issue's own HiGHS 1.14 and SCIP figures are 109 and 179 nodes, both at 0.6 s.
- The issue's side remark about OA masters and spatial B&B is not addressed (Stage 2, optional).

**2026-09-15 — swallowed exceptions in `mccormick_lp.py`.** The 14 `except Exception:`
fallbacks that were silent now log `logger.debug("<method> failed; using fallback",
exc_info=True)`. Control flow is unchanged.

**2026-09-15 — retraction: "the branch is complete".** That was said before CI ran on the
merged branch. CI then failed 14 tests in 9 files (2 lanes). They were tests whose subject is
discopt's own LP/MILP machinery and that the new default routed to HiGHS (pinned to
`DISCOPT_LP_MILP_BACKEND=rust`, none weakened), the #912 wall-budget inventory (the exact
dual correction's carved `budget` slices replaced by a deterministic work cap
`EXACT_MAX_WORK` plus the caller's own `time_limit`), and a GP-classified pure LP that the
log-space NLP answered to IPM accuracy only (now routed to HiGHS first).

**2026-09-15 — adversarial testing of the route: five soundness/robustness defects, fixed.**
A separate session ran 5,085 random LP/MILP comparisons (exact `Fraction` vertex/integer
enumeration for n ≤ 6, else scipy `milp`) plus 45 hand-built edge cases, on both backends.
Every wrong-answer claim was confirmed with an exactly checked witness.

- *F1, declared-box tightening false `infeasible` (both backends).* A ±9.999e19 box
  contribution absorbed the small terms of a row's activity sum; the leave-one-out rest came
  out 0 instead of −1 and the derived bound cut the only feasible point. Fix: widen by a
  floating-point error bound (`nonlinear_bound_tightening.py`).
- *F2, HiGHS MIP false `kInfeasible` with the finite default box.* 21 saved instances. Passing
  the sentinel-magnitude box as ±inf makes them optimal; the box-open problem is a relaxation,
  so its `kInfeasible` and tree bound stay valid.
- *F3, a proved `infeasible` overwritten by the #844 fallback.* An integer column in
  [−1.6, −1.07]; tightening proved infeasibility, the no-incumbent fallback still ran, and the
  `lp_spatial` verifier accepted a rounded-then-clipped non-integer point: `optimal`,
  certified, on both backends. Fix: the fallback skips a certified infeasible or unbounded
  result, and the verifier refuses a non-integral integer column.
- *Empty integer box with no rows.* Nothing rounded the box, and `=rust` raised
  `MILP-BB returned an infeasible point`. The tightening pass now proves infeasibility when an
  integer column's box holds no integer within the 1e-5 integrality tolerance. Fires on 0 of
  the 66 in-repo `.nl` files.
- *passModel crashes.* `kError` (row side at the 1e20 sentinel, 124 saved instances) and
  `kWarning` (HiGHS dropped |a| ≤ 1e-9, 47) raised `RuntimeError` out of `Model.solve`. Now:
  `kError` is an `error` result; on `kWarning` the model is re-passed at
  `small_matrix_value=1e-12`. Setting 1e-12 unconditionally was tried first and **falsified**:
  it turned netlib klein1 from a proved infeasible into `kUnknown`, so the option moves
  HiGHS's numerics beyond the drop threshold. A MILP still perturbed at 1e-12 is `error`
  (its label and tree bound cannot be re-derived); an LP continues, since every LP
  certificate is re-verified against the unperturbed form.

Not defects, after checking: oracle rounding and near-parallel-row mismatches (x residual
≤ 1e-8, rust agrees), two integrality cases inside the 1e-5 tolerance, and `lb > ub` refused
loudly at model build. Weak but honest, left as is: `error` where HiGHS rejects a 1e20-scale
right-hand side, and MILP `kSolveError` inside HiGHS
(`HighsMipSolverData::transformNewIntegerFeasibleSolution`).

**2026-09-15 — claim-boundary CI failure: the #1229 `dual_slack_basis` fix loosened hda.** CI on
964aea71 failed `test_current_root_lp_matches_committed_baseline`: hda root LP −64675.25 →
−5710326.49 at an identical fingerprint, same value on Linux CI and macOS.

- *Attribution* (each arm its own `CARGO_TARGET_DIR`, `.so` md5 asserted): reverting only the
  `dual_slack_basis` free-column rule restores −64675.25. The `select_leaving` and `fbbt_row`
  fixes do not move it.
- *Not unsound* (`scratchpad/hda_lp_truth.py`, every LP the relaxer solves re-solved with
  HiGHS): the branch vertex is primal feasible at −64675.25 (row violation 1.6e-10). Only its
  Neumaier–Shcherbina bound, read off inaccurate duals, is −5.71e6. Before the fix the simplex
  hit its iteration limit, the #671 failure-triggered row filter fired, and HiGHS confirms the
  filtered LP at −64675.2492.
- *Solve impact* (hda, `max_nodes=200`, TL 120, 2 interleaved rounds, both `time_limit` at 3
  nodes): main −64509.8, branch −1.40e10.
- *Entry experiment* (`scratchpad/hda_filter_entry.py`): the filtered re-solve of each of the
  branch's root LPs certifies −64675.2492.
- *Fix:* `relax_row_filter_loose_bound` (default ON, `DISCOPT_RELAX_ROW_FILTER_LOOSE_BOUND=0`
  opts out). An `optimal` node whose certified bound is more than 1e-3 relative below its vertex
  objective re-solves a row-filtered copy and keeps the larger certified bound. Sound by
  superset; inert when no row is float64-intractable.
- *Result:* root partition hda back to `unchanged` (ON: 4 unreproduced, alan/contvar/nvs08/
  tanksize, all host drift that main shows too; OFF: 5 with hda). hda solve bound −64509.8, equal
  to main. `test_row_filter_loose_bound.py` fails with the flag off (−5710326), passes on.
- *Cert-clean panel* (`scratchpad/loose_ab_guard.py`, 66 `.nl`, arms OFF/ON/ON2, TL 20 s,
  max_nodes 300): ran 66, oracle-checked arms 156, oracle violations 0, certification
  regressions 0. Changed rows: clay0303hfsg, nvs05, tls2, all noisy (ON2 differs from ON).
- The baseline is not regenerated: with the fix the committed hda row reproduces.
