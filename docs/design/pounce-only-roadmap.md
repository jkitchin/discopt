# POUNCE-Only Roadmap: Eliminating HiGHS

**Status:** Draft for review
**Scope:** Multi-phase plan to make discopt a complete, pip-installable
MILP/MIQP/MINLP solver whose only third-party numerical engine is POUNCE
(the pure-Rust Ipopt port), with HiGHS retired from the runtime dependency
set.

---

## 1. Goals and non-goals

### Goals

1. **Differentiability end to end.** Every solve path — including MILP and
   MIQP — should support gradients via implicit differentiation through the
   KKT system at the final relaxation/active set (extending T23). HiGHS
   solves break the gradient chain; POUNCE/IPM solves preserve it.
2. **A uniform batch-first stack.** One solver paradigm (interior point)
   for LP, QP, and NLP relaxations, with batch solves
   (`pounce.solve_nlp_batch`, discopt#97) as the throughput lever. No
   simplex/IPM impedance mismatch (warm starts, crossover semantics,
   tolerance mapping) between problem classes.
3. **Packaging simplicity.** `pip install discopt` ships one self-controlled
   Rust solver. No `highspy` runtime dependency, no third-party solver
   version pinning.
4. **Compete with HiGHS on pure MILP/MIQP** — as a benchmark-gated,
   directional goal (see §2), not a guaranteed parity date.

### Non-goals

- **GPU acceleration is not load-bearing** in this plan. See §4.
- We do not drop HiGHS from the *test* environment. It remains a CI
  correctness oracle through Phase 4 (§6).
- The in-house JAX IPM (T17/T24) is not being removed; its role narrows
  (§3).

---

## 2. Strategic framing

### The hard truth

Beating a mature dual-simplex branch-and-cut code on *general* MILP with an
interior-point engine is a frontier problem. Simplex's structural advantage
is millisecond warm-started re-optimization after every branch and cut —
exactly what IPMs cannot do cheaply, because an interior iterate is not on
the new central path after a bound change. No production system has
decisively beaten CPU branch-and-cut on general MILP with an IPM or
first-order stack. "Compete with HiGHS" therefore means: close the geomean
gap phase over phase, gate progress on benchmarks, and be honest that pure
MILP speed will trail for a while.

### The differentiator

discopt's unique position is not "fastest MILP." It is **the only
differentiable, batch-first, single-stack solver across MILP/MIQP/MINLP**.
A MILP solve you can take gradients through (design sensitivity,
learning-to-optimize, bilevel) is novel and valuable even at 2× HiGHS
wall-clock. Lead with that; let raw speed close over time.

### What POUNCE already gives us

Two facts confirmed in the current code make this plan viable:

- `python/discopt/solvers/nlp_pounce.py` returns **full duals** (`mult_g`,
  `mult_x_L`, `mult_x_U`). These drive reduced-cost fixing
  (`crates/discopt-core/src/presolve/duality.rs`), OBBT
  (`crates/discopt-core/src/presolve/obbt.rs`, whose plumbing is already
  waiting on an LP engine), Farkas-style infeasibility certificates, and
  the KKT point needed for implicit differentiation.
- A **batch path exists**: `_solve_batch_pounce` in `solver.py` →
  `pounce.solve_nlp_batch` (discopt#97 Phase A), validated bit-equivalent
  to serial solves in `python/tests/test_pounce_batch.py`.

### Where HiGHS is load-bearing today

"Eliminate HiGHS" is bigger than the three wrappers. Current runtime
consumers (by call-site density):

| Consumer | Role |
|---|---|
| `solver.py` | LP/QP/MILP dispatch (`_solve_lp`, `_solve_qp`, MILP hand-off) |
| `solvers/lp_highs.py`, `solvers/qp_highs.py`, `solvers/milp_highs.py` | The wrappers themselves |
| `solvers/oa.py`, `solvers/gdpopt_loa.py` | MILP master problems for OA / GDP-LOA |
| `_jax/obbt.py` | LP engine for optimality-based bound tightening |
| `_jax/mccormick_lp.py`, `_jax/milp_relaxation.py` | McCormick-LP relaxation mode |
| `_jax/partition_selection.py` | Partition-point LPs (piecewise McCormick) |
| `doe/design.py`, `doe/fractional.py`, `doe/screening.py` | MILP-based optimal design |
| `ro/formulations/polyhedral.py` | Polyhedral uncertainty subproblems |
| `_jax/convexity/linear_context.py`, `_jax/gdp_reformulate.py`, `export/`, `cli.py`, `constants.py` | Minor / incidental |

All of these must migrate to either POUNCE (continuous) or the self-hosted
integer solver (Phase 1+) before `highspy` leaves the dependency list.

---

## 3. Engine roles

Two in-house engines, one dispatch seam:

| Engine | Role |
|---|---|
| **POUNCE** (Rust) | The workhorse. All production LP/QP/NLP relaxation solves, single and CPU-batch. The only third-party-visible numerical solver. |
| **JAX IPM** (T17/T24) | The differentiable path only: custom_jvp / KKT implicit differentiation (T22/T23), and vmap-compatible evaluation kept alive for opportunistic GPU use (§4). Not a packaging dependency — it ships with discopt. |

The existing seam `python/discopt/solvers/nlp_backend.py` extends to cover
LP and QP dispatch, so no call site knows which engine is in use.

---

## 4. GPU positioning: opportunistic, not load-bearing

GPUs are **not inherently valuable** for this workload, and no phase below
gates on GPU speedups. The reasoning, recorded so we don't re-litigate it:

- **IPM cost is sparse factorization**, which is irregular and sequential —
  a poor GPU fit. POUNCE is CPU Rust by design.
- **The B&B frontier is narrow when it matters.** Good pruning keeps the
  open-node set small (8–32 nodes); at those batch widths a multicore CPU
  batch matches or beats a GPU after launch/transfer/compile overhead.
- **Target problems are small** (phase gates at ≤100 variables). Batched
  dense factorizations at that size are microsecond-scale on CPU; GPUs win
  at sustained batch widths in the thousands, which B&B rarely supplies.
- **First-order GPU LP (cuPDLP-class) is the wrong regime**: it wins on
  LPs with millions of nonzeros at moderate accuracy; B&B dual bounds need
  high-accuracy solves of small LPs.

Where GPU *does* genuinely help, we keep the option open at zero roadmap
cost:

- Batch McCormick/relaxation **evaluation** (pure vmapped function
  evaluation — how the Phase 2 "GPU speedup ≥15×" gate was met on pooling).
- Naturally wide solve waves: OBBT (2n LPs/round), multistart, batched
  strong branching, diving — though these parallelize fine on CPU cores.
- **NN-embedded MINLP** (`discopt.nn`): dense matmul-dominated relaxations
  are genuinely GPU-shaped. If NN-constrained optimization becomes a
  flagship use case, revisit GPU then.

Policy: keep relaxation code vmap-compatible (cheap to maintain), deliver
batch width via multicore Rust, treat GPU as upside.

Note: "batch width replaces warm-starting" is the IPM-world translation of
simplex's advantage. Interior-point warm-starting is weak by nature (a
parent's interior iterate is off the child's central path), so the current
multistart-from-clipped-parent-solution strategy in `solver.py` is correct,
not a workaround. Do **not** build LP-basis warm-start.

---

## 5. The crossover fork (decided: build it)

To compete on MILP we need the workhorse general-purpose cuts — Gomory
mixed-integer, MIR, lift-and-project — and those are **basis-derived**. An
IPM returns an interior point, not a vertex with a basis. Decision:

**Build a small pure-Rust crossover** (interior point → vertex + basis),
run it at the **root and periodically** — never per node — generate basis
cuts once, add them to the formulation, then resume batch IPM solving.

Rationale: it is "our own small simplex," so it serves packaging and
uniformity while unlocking the strongest general MILP cuts. Per-node
crossover would reintroduce the IPM anti-pattern (sequential cut-and-resolve
loops). Fallback if crossover stalls numerically: basis-free cut families
only (cover, clique, flow-cover, implied-bound, RLT, conflict), accepting a
harder road to MILP parity.

---

## 6. Phases

| Phase | Goal | HiGHS sites retired | Exit gate |
|---|---|---|---|
| **0** | POUNCE as universal continuous engine | `lp_highs`/`qp_highs` internals | LP/QP match HiGHS to tolerance on `test_lp_qp_solvers`; duals + Farkas ray reliable; bound trust-gate in place |
| **1** | Self-hosted integer B&B (cover) | `milp_highs` delegation | `incorrect_count = 0` on all MILP/MIQP suites |
| **2** | Crossover + root cuts (compete I) | — | Measurable root-gap reduction on MIPLIB subset; zero invalid cuts |
| **3** | Cut & heuristic suite (compete II) | — | Material node-count reduction vs Phase 1; presolve clique table consumed |
| **4** | Retire remaining HiGHS consumers | `oa.py`, `gdpopt_loa.py`, `obbt.py`, `mccormick_lp.py`, `partition_selection`, DOE, RO, export/cli | `highspy` out of runtime deps; HiGHS import in tests only |
| **5** | Parity push + differentiability completion | — | Geomean vs HiGHS within target; gradients through MILP/MIQP solves validated |

### Phase 0 — POUNCE as the universal continuous engine

Foundation; correctness-critical. Nothing downstream is safe until POUNCE
is a trustworthy LP/dual engine. Work breakdown:

- **P0.1 Pure-LP path — DONE (validation).** `discopt/solvers/lp_pounce.py`
  provides `solve_lp(c, A_ub, b_ub, A_eq, b_eq, bounds, ...)` —
  signature/return-type compatible with `lp_highs.solve_lp` — by exposing the
  LP to POUNCE as a zero-Hessian, constant-Jacobian callback problem.
  `test_lp_pounce.py` mirrors `test_lp_highs.py` and cross-checks objectives
  against HiGHS (P0.5 oracle). Validation outcomes:
  - Objective matches HiGHS across inequality / equality / mixed / bounded /
    free-variable / empty-constraint LPs and a random battery.
  - Convex corner cases map soundly: inequality-infeasible → INFEASIBLE,
    unbounded → UNBOUNDED (raw Ipopt code 4, which the *general* NLP map sends
    to ERROR — hence the LP-specific status map).
  - Degenerate / dual-degenerate LPs return the analytic center of the
    optimal face, not a vertex: objective matches, primal differs by design.
  - **Known limitation:** an *inconsistent-equality* system can cycle to the
    iteration limit instead of reporting INFEASIBLE (the IPM's restoration
    doesn't always certify infeasibility). Result is still sound (never a
    bogus OPTIMAL, no solution returned); clean certification needs the
    Farkas-ray check from P0.2.
  - No simplex basis / warm-start: `LPResult.basis` is `None`, `warm_basis`
    is accepted but ignored (an IPM does not warm-start from a basis).
  - *Not yet wired in:* downstream consumers (OBBT, McCormick-LP, masters)
    and `_solve_lp` still call HiGHS; switching them over is P0.4/P4. Sparse
    `A` is densified for now (validation-scale); sparse Jacobian is a P0.4
    follow-up for large LPs.
- **P0.2 Infeasible-node handling.** The cyipopt status map inherited by
  `nlp_pounce.py` sends Ipopt status 2 (*locally* infeasible) to `ERROR`.
  For an IPM this means "this start found no feasible interior point," not
  "the node is infeasible" — pruning on it is unsound. Required behavior:
  feasibility-restoration retry / restart on local infeasibility; declare a
  node infeasible **only** on a genuine Farkas dual ray (infeasibility
  certificate). Expose the certificate through `NLPResult`.

  *Status / implementation notes:*
  - **LP infeasibility certificate — DONE.** `lp_pounce.solve_lp` now
    disambiguates a non-optimal IPM exit (ITERATION_LIMIT / ERROR) with an
    elastic Phase-1 LP that minimizes total constraint violation
    (`_phase1_min_violation`). By LP duality a positive minimal violation is a
    Farkas certificate, so the result is reported as INFEASIBLE; a ~0 minimum
    proves feasibility (the failure was numerical, reported honestly).
    Closes the P0.1 inconsistent-equality limitation
    (`test_lp_pounce.py::TestInfeasibilityCertificate`), and the Phase-1 path
    only runs on non-optimal exits so feasible/optimal solves are untouched.
  - **Nonconvex B&B nodes — by design, no NLP-based certificate.** Local
    infeasibility of a nonconvex node's NLP relaxation is *not* a global
    infeasibility proof, so the node solve must never rigorously prune a
    nonconvex node; only FBBT (interval) infeasibility is a certificate
    there. The soundness half of this is already handled (commit 3b59e8b:
    a failed/locally-infeasible node carries a sentinel that decertifies the
    gap rather than silently pruning).
  - **Expose the certificate — DONE.** `InfeasibilityCertificate`
    (`total_violation`, `ineq_violations`, `eq_violations`; the Phase-1
    per-row slacks) is attached to `LPResult.infeasibility_certificate` on an
    infeasible result — for free on the disambiguation path, and on demand
    (`solve_lp(certificate=True)`) for a directly POUNCE-detected
    infeasibility. `SolveResult.infeasibility_certificate` surfaces it at the
    model level (the POUNCE LP engine requests it). Note: at full
    `Model.solve` level FBBT often proves simple infeasible LPs *before* any
    engine runs (no certificate there); the witness is most useful to the
    internal LP consumers (OBBT, masters) that call `lp_pounce.solve_lp`
    directly. Tests: `TestInfeasibilityCertificate`,
    `TestInfeasibilityCertificateExposed`.
  - **Serial POUNCE node retry — DONE.** `_solve_node_nlp_pounce` now retries
    a failed solve (error / divergence / local infeasibility) from up to two
    alternative deterministic starts (box midpoint, then an off-center point
    at lb + 0.382·width — no RNG, determinism by default) before reporting
    the failure. Without this a single bad start cost the node its bound and
    decertified the gap. Tests: `test_p03_trust_gate.py::TestSerialNodeRetry`
    (retry on failure, three-attempts-then-give-up, no retry on success).
    P0.2 is complete.
- **P0.3 Bound trust-gate.** Use a POUNCE objective as a valid dual bound,
  and its multipliers for reduced-cost fixing, **only** when the solve
  converged to tolerance. An unconverged IPM bound used for fathoming can
  cut the optimum — this violates the `incorrect_count ≤ 0` invariant.

  *Status / implementation notes (from code investigation):*
  - **Batch sentinel parity — DONE (commit 3b59e8b).** The batch IPM/POUNCE
    path used to silently prune nodes whose relaxation solve *failed*
    (error, divergence, local infeasibility) and still report a certified
    `optimal`. It now mirrors the serial path's #27a guard: such nodes
    decertify the gap, and the McCormick-LP bound rescues them where the
    relaxer is active. Verified that all four paths (pounce/ipm ×
    batch/serial) now agree (e.g. Haverly pooling → `feasible`, obj 1390).
  - **Convex non-KKT objectives — DONE (trusted-mask decoupling).** For a
    *convex* model the node relaxation objective is the node lower bound; a
    non-KKT result (Ipopt `ITERATION_LIMIT`, i.e. JAX-IPM codes 3/4 or a
    POUNCE max-iter exit) gives `f(x~) ≥ f*`, not a valid lower bound (#39).
    Resolved without touching bound/incumbent (avoiding the conflation
    below): `_solve_batch_ipm`/`_solve_batch_pounce` now return an additive
    5th value, a per-node `trusted` mask — `False` for a convex node whose
    objective is non-KKT and could not be polished to optimality (JAX-IPM
    polishes via `_solve_node_nlp_kkt`/POUNCE; POUNCE has no polish so any
    non-optimal-but-usable convex result is untrusted). Both B&B loops
    (`solve_model`, `_solve_nlp_bb`) and the serial paths decertify the gap
    (`_gap_certified = False`) on untrusted/`ITERATION_LIMIT` convex nodes
    and report `"feasible"` rather than a falsely-certified `"optimal"` —
    bounds and incumbent are left untouched. `_solve_nlp_bb` additionally now
    gates `"optimal"` on `_gap_certified` (it previously reported `"optimal"`
    on search closure alone, also unsound for nonconvex heuristic-mode runs)
    and nulls the bound/gap when uncertified. Tests:
    `test_p03_trust_gate.py` (mask both solvers + caller decertification,
    batch & serial). Nonconvex models are unaffected — they discard the NLP
    objective (`trusted` stays `True`).
  - **Why not `-inf`/sentinel (the conflation that forced the mask).** In
    *convex* mode the Rust tree uses `node_lb` as **both** the dual bound and
    the incumbent objective for integer-feasible nodes (`process_evaluated`,
    `tree_manager.rs`): `-inf` would corrupt the incumbent, a sentinel would
    prune a node we failed to bound. The `trusted` mask sidesteps this by
    leaving bound values intact and only decertifying the gap.
  - **POUNCE polish-retry — DONE.** A convex node stalling at
    `ITERATION_LIMIT` now gets one boosted re-solve (max(3×`max_iter`, 3000)
    iterations, warm-started from the stalled iterate) before trust is
    withheld: in `_solve_node_nlp_pounce` (serial, via `convex=` forwarded
    from `_solve_node_nlp`) and in `_solve_batch_pounce`'s reduce loop and
    serial fallback. Only an OPTIMAL (KKT) polish restores the bound/trust;
    otherwise the node decertifies as before. Recovers nodes that previously
    just decertified — e.g. a `max_iter=1` stall now returns the true
    optimum with `trusted=True`
    (`TestTrustedMaskPOUNCE::test_stalled_convex_is_rescued_by_polish_retry`).
- **P0.4 Batch LP/QP + LP seam.** Extend the `solve_nlp_batch` path to LP/QP
  node waves; extend the backend seam so LP/QP dispatch is engine-agnostic.
  Note: the published `pounce-solver` wheel (0.4.0) exposes
  `pounce.Problem`/`.solve` but **not** `solve_nlp_batch`, so
  `_solve_batch_pounce` currently falls back to serial per-node solves;
  batch LP/QP depends on that POUNCE API landing.

  *Status / implementation notes:*
  - **LP seam — DONE.** `_solve_lp` now tries matrix-form engines in order
    HiGHS → POUNCE → JAX IPM, flipped to POUNCE-first on
    `nlp_solver="pounce"` (the user asked for POUNCE everywhere). The shared
    model→matrix→SolveResult body is factored into `_solve_lp_matrix`;
    `_solve_lp_highs` / `_solve_lp_pounce` are thin engine wrappers. Default
    behavior unchanged (HiGHS first; `nlp_solver` defaults to `"ipm"` with no
    auto-resolution). Routing pinned by `test_lp_backend_seam.py`.
  - **Dual convention reconciled.** Ipopt-style `mult_g` enters the
    Lagrangian as `f + mult_g^T g` — the negation of the HiGHS shadow-price
    convention `LPResult` documents. `lp_pounce` negates row duals so both
    backends agree (exact parity tested on unique-dual LPs;
    `TestDualConvention`). Reduced costs (`mult_x_L - mult_x_U = c - A^T y`)
    already matched. On dual-degenerate LPs the IPM returns an interior
    point of the dual optimal face — valid, but not simplex's vertex dual.
  - **QP seam — DONE.** `_solve_qp` mirrors the LP seam: HiGHS → POUNCE →
    JAX QP IPM, flipped to POUNCE-first on `nlp_solver="pounce"`. The shared
    body is `_solve_qp_matrix`; `discopt/solvers/qp_pounce.py` provides the
    matrix-form engine (objective `0.5 xᵀQx + cᵀx`, constant Jacobian/Hessian
    callbacks; reuses lp_pounce's status map, dual-sign reconciliation, and
    elastic Phase-1 certificate — Phase-1 is objective-independent, so it
    applies to QPs unchanged). The POUNCE engine **declines integrality**
    (returns None for MIQPs — no B&B in an IPM), keeping MIQPs on
    HiGHS / the B&B path. `QPResult` gains `infeasibility_certificate`.
    Tests: `test_qp_pounce.py`, `test_qp_backend_seam.py`.
  - **Engine-result feasibility guard.** Validation surfaced a real HiGHS QP
    failure: on a small random strictly convex QP (`default_rng(1)`, n=4,
    m=5, box ±5) HiGHS returns a point violating its constraints by 7.5
    labeled `kOptimal` (obj 83.18 vs the true 1.672, POUNCE- and
    SLSQP-confirmed; 1 of 12 seeds). `_solve_lp_matrix`/`_solve_qp_matrix`
    now verify the returned point against its own constraints
    (`_matrix_solution_feasible`) and fall through to the next engine on
    violation — no engine's "optimal" is taken on faith (P0.5 in both
    directions: the oracle itself can lie).
  - **Open:** batch LP/QP waves (blocked on POUNCE `solve_nlp_batch`),
    OBBT/McCormick-LP consumers (Phase 4).
- **P0.5 HiGHS as CI oracle.** From this phase on, cross-check every POUNCE
  LP/QP result against HiGHS in CI (test-only dependency). HiGHS stops
  being a runtime engine long before it stops being a correctness guard.

### Phase 1 — Self-hosted integer B&B ("cover" milestone)

*Status / implementation notes:*
- **Scoping finding.** `_solve_milp_bb` / `_solve_miqp_bb` already exist as
  self-hosted Rust-tree B&B with **in-house JAX LP/QP IPM** node relaxations
  (no third-party solver — already POUNCE-only-compatible); they are the
  *fallback* today (HiGHS is default via `use_highs_milp=True`). "Cover" =
  make them sound, then default.
- **Correctness floor — DONE.** Both accepted a non-KKT (max-iter,
  `converged==3`) relaxation objective as a valid lower bound and certified
  `"optimal"` on `tree.gap()`/`is_finished()` alone — a latent wrong-`optimal`
  path (the same #39 class as P0.3). Fixed: a code-3 node bound decertifies
  the gap (`_gap_certified=False`), `"optimal"` now requires a closed search
  **and** a certified gap, the bound/gap are nulled when uncertified, and
  `SolveResult.gap_certified` is surfaced. Bounds/incumbent untouched.
  Tests: `test_p1_milp_bb_soundness.py` (controls still certify; forced
  code-3 → `"feasible"`, `gap_certified=False`, correct incumbent).
- **POUNCE recovery for stalled nodes — DONE (increment 2).** Instead of
  decertifying on a code-3 node, `_pounce_recover_node_bound` re-solves the
  node in original-variable matrix form via `lp_pounce`/`qp_pounce`: an
  OPTIMAL result is a KKT-valid bound (node recovered, search certifies
  normally), INFEASIBLE is Phase-1-certified (rigorous prune); only when
  POUNCE is unavailable or also fails does the gap decertify. Wired into the
  batch and serial paths of both functions.
- **Interior-solution purification — DONE (increment 3).** IPM relaxation
  optima are interior, so integer coordinates come back smeared (e.g.
  0.99996) beyond the 1e-5 integrality tolerance. `_pounce_snap_incumbent`
  rounds integer coordinates within 1e-4, validates them against the
  original node box, *fixes* them, and re-solves the continuous relaxation
  with POUNCE — yielding an exactly feasible integer point with an exact
  objective, injected via `tree.inject_incumbent` (also a rounding
  heuristic). Visible win: the knapsack MILP incumbent is exactly `-8.0`
  instead of the smeared `-7.999999997754972`.
- **Reduced-cost fixing — DONE (increment 4).** `_root_reduced_cost_fixing`
  solves the root LP with POUNCE (KKT-valid duals), purifies a near-integral
  point into an incumbent, and tightens integer bounds via
  `_reduced_cost_fixing`: for a minimization relaxation with lower bound
  `z_lp`, reduced costs `d`, and incumbent `z_inc`, each non-basic integer is
  capped by `d_j·(x_j − bound_j) ≤ z_inc − z_lp`. The true optimum (objective
  `≤ z_inc`) always satisfies these, so RCF never cuts it; `gap` is inflated
  by a small relative margin so interior-point dual tolerance cannot
  over-tighten. Best-effort and graceful (only tightens; skipped if POUNCE is
  absent or no incumbent recoverable), wired at the root of `_solve_milp_bb`
  before tree creation. A 200-case property test confirms no integer point
  with objective `≤ z_inc` is ever excluded; an end-to-end test confirms the
  answer is identical with and without RCF. (RCF is an LP technique — the
  quadratic term breaks the reduced-cost decomposition — so it is MILP-only;
  OBBT and MIQP bound-tightening remain optional perf follow-ups.)
  - *Note:* purification (increment 3) can change which of several tied
    optima becomes the incumbent; `test_solver_duals` was made well-posed
    (unique optimum) so its recovered-dual assertion no longer depends on the
    tie-break.
- **Flip off HiGHS — DONE (increment 5).** With `nlp_solver="pounce"` (the
  POUNCE-only opt-in, mirroring the LP/QP seams), MILP/MIQP now route
  straight to the self-hosted B&B and bypass HiGHS entirely: the HiGHS
  pre-try is skipped and the incumbent dual recovery
  (`_mip_recover_relaxation_duals`) re-solves the fixed relaxation via
  `lp_pounce`/`qp_pounce` (`prefer_pounce` threaded through both B&B
  functions). Verified HiGHS-free: with every HiGHS entry point made to
  raise, `m.solve(nlp_solver="pounce")` solves the knapsack MILP (−8.0,
  certified, duals recovered) and the MIQP (0.18) correctly; the default
  mode still tries HiGHS first (routing unchanged). The *global* default
  flip — removing HiGHS for non-pounce users — is deferred to Phase 4 when
  HiGHS is dropped as a dependency; flipping a slower engine on for everyone
  is out of scope for the "cover" milestone.

**Phase 1 complete (cover milestone):** the self-hosted MILP/MIQP B&B is
sound (incr 1), recovers stalled nodes via POUNCE (2), purifies incumbents
(3), reduced-cost-fixes at the root (4, MILP), and is fully HiGHS-free in
POUNCE-only mode (5). Remaining MILP work (cutting planes, crossover,
competitive performance) lives in Phases 2–3.

### Phase 3 — Cut & heuristic suite (started)

- **Knapsack cover cuts — DONE.** `_jax/cover_cuts.py` separates valid cover
  inequalities `sum_{j in C} x_j <= |C|-1` from binary-knapsack rows; a root
  cut loop (`_root_cover_cut_loop`) solves the root LP, separates violated
  covers, and augments the standard-form `lp_data` with each cut as a row +
  non-negative slack (`_augment_lpdata_with_cover_cuts`) — tightening every
  node's relaxation without touching the tree's variable structure. Cuts are
  rigorously valid (exhaustively verified to never exclude a feasible 0/1
  point), so the optimum is preserved. On a unique-optimum knapsack the node
  count dropped 15 → 7. Basis-free, so it composes with the IPM relaxations.
  - **Empirical finding (motivates the Phase-2 crossover).** Cover cuts
    separate a *vertex* sharply but an interior point weakly: on a symmetric
    knapsack the IPM returns the analytic center `[0.45]*4`, which violates no
    cover (`0.9 < 1`), whereas simplex's vertex `[1, 0.8, 0, 0]` is cut
    immediately. Effective MILP cutting on this stack ultimately wants the
    IPM→vertex crossover (Phase 2 keystone).
- **Clique cuts — DONE.** `_extract_clique_edges` runs the Rust presolve
  conflict-graph pass; `separate_clique_cuts` greedily *merges* the 2-clique
  edges into larger cliques and emits the violated `sum_{j in C} x_j <= 1`
  (folded into `_root_cover_cut_loop`). Pairwise edges are usually redundant
  with their source constraints, but a merged clique of size >= 3 separates
  even the symmetric IPM center (`0.5*3 > 1`) — partly overcoming the
  interior-point cut limitation that defeats cover cuts. Exhaustively
  verified valid (cuts are true cliques, never exclude a feasible point); a
  triangle set-packing drops 3 → 1 nodes. (`test_clique_cuts.py`)
- **Root diving heuristic — DONE.** `_root_dive` fixes the most-fractional
  integer and re-solves the LP until integral, injecting an early incumbent
  that front-loads pruning / reduced-cost fixing (complements the
  near-integral snap purification). Optimum preserved, node count never
  worsened. (`test_root_dive.py`)
- **IPM→vertex crossover — DONE (Python/numpy).** `_jax/crossover.py`
  (`crossover_to_vertex`) pushes an interior LP optimum to a vertex of the
  optimal face: it repeatedly moves along a null-space direction supported on
  the *free* variables with `A d = 0` and `c^T d = 0` (SVD-based, so both
  feasibility and objective are preserved exactly) and ratio-tests to the next
  bound, fixing one variable per step until no free direction remains —
  terminating in at most `n` steps. The root cut loop now separates from the
  crossed-over vertex instead of the raw interior point, so cover/clique cuts
  finally bite from an IPM solve: on the symmetric knapsack the analytic center
  `[0.45]*4` violates no cover, but its crossover vertex `[1, 0, 0.8, 0]` is
  cut immediately, dropping the node count 21 → 1 with the optimum preserved.
  Soundness is unaffected by crossover numerics — the crossover only *locates*
  candidate cuts; each cut is still validated by its own cover/clique structure.
  A size guard (`_MAX_CROSSOVER_VARS = 400`) falls back to interior-point
  separation on very wide problems. A pure-Rust port (with basis recovery, the
  prerequisite for Gomory/MIR) remains future work. (`test_crossover.py`)
- **Open (Phase 2 keystone, Rust):** the Rust crossover port with *basis*
  recovery, the path to basis-derived Gomory/MIR cuts. Also: RINS (sub-MILP
  neighborhood search) and conflict analysis. These are the remaining
  "compete" items; the tractable Python-side cut/heuristic pieces are now done.

### Phase 4 — Retire remaining HiGHS consumers (in progress)

Goal: discopt runs fully with **only POUNCE installed** (no HiGHS). Each
matrix-form HiGHS caller selects a signature-compatible engine through a
shared seam and falls back to whichever backend is importable.

- **LP/QP backend selector — DONE.** `discopt/solvers/lp_backend.py`
  (`get_lp_solver`/`get_qp_solver`, `prefer_pounce=`) picks HiGHS or POUNCE,
  HiGHS-first by default and POUNCE-first in POUNCE-only mode, falling back to
  whichever is importable (raises only if neither is). Tested incl. a
  simulated HiGHS-absent install.
- **OBBT — DONE.** `_jax/obbt.py` (`run_obbt`, `run_obbt_on_relaxation`) now
  goes through the selector with a `prefer_pounce` param, threaded from the
  spatial-B&B OBBT call site (`nlp_solver="pounce"`). Verified it tightens
  identically across backends and works with HiGHS absent.
- **Matrix-form MILP via self-hosted B&B — DONE.** `solvers/milp_pounce.py`
  exposes the Phase-1 B&B behind the `milp_highs.solve_milp`
  signature/`MILPResult` contract by building a `Model` from the matrices and
  running `_solve_milp_bb(prefer_pounce=True)`, then mapping back
  (`"optimal"→OPTIMAL`, `"feasible"/"node_limit"→ITERATION_LIMIT+incumbent`,
  etc.). `get_milp_solver` added to the selector. The three matrix-MILP
  consumers — the **OA and GDP-LOA convex-MINLP masters** and
  `milp_relaxation` — now go through it, so they run with only POUNCE. The
  Model round-trip is validated faithful: the adapter matches HiGHS on a
  12-seed random MILP battery (0 mismatches), and OA's master solves
  HiGHS-free end to end (`test_milp_pounce.py`,
  `test_lp_backend_select.py::test_oa_master_is_highs_free`).
- **Raw-`highspy` consumers retired — DONE.** The three remaining call sites
  that imported a HiGHS wrapper (or `highspy`) directly now go through the
  selector and so run with only POUNCE installed:
  - **`partition_selection._solve_vertex_cover_milp`** — the covering MILP
    (binary `min Σy s.t. Ay ≥ 1`) built by hand against `highspy.Highs()` now
    calls `get_milp_solver()` and keeps its greedy-cover fallback.
  - **`solver._strong_branch_lp`** — strong-branching's up/down LP probes take
    a `prefer_pounce` flag (threaded from `nlp_solver == "pounce"` at the call
    site) and select through `get_lp_solver`; if no LP backend is importable
    it returns `None` and the tree falls back to pseudocost branching.
  - **`gdp_reformulate._compute_big_m_lp`** — multiple-big-M's LP-based bound
    tightening selects through `get_lp_solver()` (keeping its interval-
    arithmetic fallback).
  Verified HiGHS-free and (for big-M) numerically identical to HiGHS in
  `test_lp_backend_select.py::TestHighspyConsumersRetired`. The DOE and
  robust-opt paths the earlier scan flagged were already clean: both build a
  `Model` and call `Model.solve()`, which is POUNCE-capable. No remaining
  module imports `highspy` outside the `*_highs.py` wrappers themselves, and
  those are imported only lazily through the selector — so a `[pounce]`-only
  install is fully functional. `highspy` is already an optional extra (`highs`,
  and in `dev`), and we keep it that way: HiGHS stays the default engine when
  installed and the reference oracle (see the Phase 4 packaging note).

- Route pure MILP and MIQP through the existing Rust B&B
  (`crates/discopt-core/src/bnb/`) with POUNCE LP/QP relaxations, replacing
  the `milp_highs.py` whole-problem hand-off.
- **Purification of interior solutions.** IPM optima are interior, so
  integer variables come back smeared (many slightly-fractional values),
  making integer-feasibility detection and most-fractional branching noisy.
  Add lightweight purification (push toward a vertex, or tolerance-aware
  rounding with feasibility repair) as the cheap interim before Phase 2
  crossover.
- **Switch on dual-driven reductions.** Reduced-cost fixing (`duality.rs`)
  and OBBT (`obbt.rs`) driven by POUNCE multipliers — both already plumbed
  on the Rust side and waiting on exactly this.
- Convex/indefinite split in `_solve_miqp_bb`: convex Q gets the direct QP
  fast path; indefinite Q gets spatial branching + eigenvalue-based
  convexification.
- Hard gate: `incorrect_count = 0` across MILP/MIQP benchmark suites.
  Correctness before speed; slower than HiGHS is acceptable here.

### Phase 2 — Crossover + root cut generation (compete I)

- **Primal crossover (interior → vertex) — DONE in Python.**
  `_jax/crossover.py`; null-space push preserving objective and feasibility,
  wired into the root cut loop so cover/clique cuts bite from IPM solves
  (symmetric knapsack 21 → 1 nodes). See the Phase 3 (started) section for
  details.
- **Pure-Rust crossover** (interior → basic feasible vertex + *basis*).
  Remaining keystone: the basis (not just the vertex) is the prerequisite for
  basis-derived Gomory/MIR cuts. Use HiGHS basis output as the test oracle.
  - **Increment 1 — vertex crossover core — DONE.**
    `crates/discopt-core/src/lp/{mod,crossover}.rs`. New dependency-free `lp`
    module: `crossover_to_vertex(x, &LpView, tol, max_iter)` ports the Python
    null-space push to Rust — `LpView` is a borrowed standard-form LP view, the
    null direction of `[A_free; c_freeᵀ]` comes from a hand-written
    rank-revealing RREF (no BLAS/LAPACK dep, keeping the wheel build clean),
    and a ratio test fixes one variable per step until the free columns are
    independent (a vertex). 5 Rust unit tests (objective + feasibility
    preserved, lands on a vertex, already-vertex stable, size guard, 50 random
    LPs); `cargo test/clippy/fmt` clean.
  - **Increment 2 — basis recovery — DONE.** `lp/basis.rs`:
    `recover_basis(x, &LpView, tol) -> Option<Basis>` turns a vertex into a
    simplex basis — the free (strictly-interior) variables must be basic, then
    the basis is completed greedily with at-bound columns that raise the rank
    of `A_B` (incremental Gaussian elimination via a `RankTracker`), and the
    rest are classified `AtLower`/`AtUpper` with HiGHS-compatible status codes
    (`kLower=0`/`kBasic=1`/`kUpper=2`). It **declines** (returns `None`) on a
    point that is a vertex of a higher-dimensional optimal face but *not* a
    polytope vertex (> `m` interior vars) — basis recovery is only well-defined
    at an actual basic feasible solution, i.e. an LP optimum after crossover.
    On a degenerate vertex many bases are valid; recovery returns *a* valid one
    (the soundness property cuts need), not necessarily simplex's. 4 Rust tests
    assert basis *validity* — `|B| = m`, free vars basic, nonbasics at their
    bound, and `A_B x_B = b − A_N x_N` reconstructs the vertex — over a clean
    vertex, a vertex with a free (basic) variable, an end-to-end
    crossover→recover, and the declined non-vertex case.
  - **Increment 3 — PyO3 binding — DONE.**
    `crates/discopt-python/src/lp_bindings.rs` exposes
    `discopt._rust.crossover_to_vertex_py(x, a, c, lb, ub, tol, max_iter)` and
    `recover_basis_py(x, a, c, lb, ub, tol) -> (col_status, basic_vars) | None`,
    taking the standard-form LP as zero-copy numpy (C-contiguous `A`). The
    Python B&B can now call the Rust crossover and consume the basis.
    `test_rust_crossover.py` drives the real pipeline (`extract_lp_data` →
    `lp_ipm_solve` interior optimum → Rust crossover → basis): objective and
    feasibility preserved and lands on a vertex; matches the numpy reference's
    *properties* (both reach a valid vertex, tie-breaking aside); the recovered
    basis is valid (reconstructs the vertex); and it reproduces HiGHS's
    optimum. (CI builds the extension with maturin, so the bindings are present
    there; the test skips if an older prebuilt `.so` lacks them.)
  - **Increment 4a — Gomory cut generation core — DONE.** `lp/gomory.rs`:
    `separate_gomory(&LpView, &Basis, integrality, x, tol, max_dynamism)`
    derives a Gomory mixed-integer (GMI) cut from each fractional integer basic
    variable's tableau row. For basic var `B_i` it solves `Bᵀ w = e_i` to get
    row `i` of `B⁻¹`, forms `ā_j = w·A_j` for nonbasic `j`, works in the shifted
    nonbasic space `x̃_j ≥ 0` (sign-flipping at upper bounds via the recovered
    `col_status`), applies the standard GMI formula, and maps the resulting
    `Σ ψ_j x̃_j ≥ 1` back to a cut `Σ γ_j x_j ≥ δ` over the original variables.
    Numerically guarded (fractionality tolerance, a max-dynamism filter, finite
    coefficients). The cut is valid for *any* basis row with an integer basic
    variable — optimality not required — so soundness is independent of
    crossover/IPM numerics. 3 Rust tests verify, by enumerating all
    integer-feasible points, that each cut excludes none of them yet cuts off
    the fractional vertex (worked example: `x0+x1+s=1.5` at `(1, 0.5, 0)` →
    `2s ≥ 1` ⇔ `x0+x1 ≤ 1`), plus a two-constraint case and the no-cut
    (integral) case.
  - **Increment 4b — PyO3 binding — DONE.** `lp_bindings.rs` adds
    `discopt._rust.gomory_cuts_py(x, a, c, lb, ub, integrality, tol,
    max_dynamism)`, which recovers the basis at `x` and separates GMI cuts in
    one call, returning `(coeffs[k×n], rhs[k])` (cuts `coeffs[i]·x ≥ rhs[i]`) or
    `None` when `x` is not a basic feasible solution. `test_rust_crossover.py`
    adds the worked `2s ≥ 1` example (exact coefficients + validity by
    enumeration) and a real-pipeline test (IPM → crossover → GMI) asserting the
    cuts are finite and separate the fractional vertex.
  - **Increment 5 — wire into the root cut loop — ATTEMPTED, REVERTED (not
    yet safe).** Wired `gomory_cuts_py` into `_root_cover_cut_loop` (separate at
    the crossover vertex, augment `lp_data` with `coeffs·x − s = rhs` rows under
    a safe-GMI rhs margin). The `test_milp_pounce` HiGHS-agreement battery
    immediately caught a **correctness violation**: on a general-integer seed
    the solver returned `-11` labelled optimal vs HiGHS's true `-13` — a GMI cut
    excluded the true integer optimum. Root cause: GMI is derived from the
    *numerical* IPM→crossover basis, and the tableau coefficient `ā_j = w·A_j`
    inherits the basis' error; near an integer it flips the fractional part
    `f_j` between ≈0 and ≈1, changing a coefficient by O(1). The fixed rhs
    margin cannot absorb this because the cut-value error at an integer point
    scales with that point's *distance from its bound* — unbounded for
    general-integer variables (the failing case had a variable at value 2).
    Wiring reverted to keep `incorrect_count == 0`.
  - **Increment 5b — basis refinement (DONE) + wiring (still blocked, deeper
    cause found).** Chose option (b): `lp/gomory.rs` now reconstructs the vertex
    and tableau from the *exact* basis and bounds with **iterative refinement**
    (`solve_refined`) instead of trusting the IPM vertex, snaps the refined
    `ā_j` to integers within `SNAP_TOL`, and filters on a fractionality band
    (`FRAC_MIN`) and an absolute coefficient cap (`MAX_ABS_COEFF`).
    `separate_gomory` now takes `b` (the rhs) and needs no input vertex.
    **This fixed cut accuracy**: a dedicated general-integer-at-upper-bound unit
    test (variable at value 2 — the failing class) passes, and with refinement
    the full `test_milp_pounce` HiGHS battery passes *with GMI wired*, with node
    reductions on several seeds (3→1).
    **But a third, deeper failure surfaced and the wiring was reverted again.**
    On a pure-binary knapsack the solver returned `-8` vs the true `-10` even
    though every generated cut was valid and the optimum stayed LP-feasible in
    the augmented system. Cause: the GMI cut came out over the original `≤`-row
    *slack* (`0.25·x₄ ≥ 1`, where `x₄ ∈ [0, 1e20]`); **stacking that
    huge-range-slack-coupled cut with the cover cuts makes the augmented LP
    ill-conditioned and the IPM diverges** (`converged=3`, `obj=nan`), so the
    B&B prunes on garbage bounds. Cover-cuts-only converges; a single GMI cut
    converges; the *combination* does not.
    Diagnosis: the blocker was no longer cut accuracy (refinement solved that)
    but **IPM robustness on cut-augmented relaxations**.
  - **Increment 5c — structural projection — DONE (GMI now wired and
    correct).** `_project_cut_to_structural` (solver.py) substitutes every
    slack `x_s = (b_r − Σ_k A[r,k] x_k)/c_s` via its singleton defining row,
    turning a slack-coupled cut like `0.25·x₄ ≥ 1` into the structural
    `−1.25·Σx ≥ −1.25` (i.e. `Σx ≤ 1`): O(1) coefficients, no coupling to the
    `1e20`-range row slacks. The substitution is exact through `A_eq x = b_eq`
    (true at every feasible point), so validity and vertex-separation are
    preserved. `_separate_gomory_cuts` now projects each refined GMI cut before
    augmentation, and the root loop runs GMI on round 0 only. **Results:** the
    pure-binary knapsack that returned `-8` now returns the correct `-10` at the
    root; the full `test_milp_pounce` HiGHS battery passes with GMI wired;
    objectives stay consistent and GMI reduces the node count on ~⅓ of the
    battery seeds (never worsens). New tests in `test_rust_crossover.py`
    (`TestGomoryWiring`): the projection unit test and an end-to-end correctness
    + node-reduction test on the formerly-failing knapsack. **GMI cuts are now a
    live, sound part of the MILP solve path** — the payoff of the crossover +
    basis keystone. The `test_milp_pounce` battery + the cover-cut end-to-end
    test remain the standing correctness safeguards.
  - **Increment 5d — opt-in by default (performance).** Always-on GMI
    regressed the full fast suite ~30% (≈11→14 min): in this JAX stack, adding
    cut rows changes the LP shape and forces the interior-point solver to
    **recompile** for the augmented problem. Cover/clique cuts avoid this
    because they only fire on knapsack/clique problems; GMI hitting *every*
    integer problem made all of them pay the recompile + an extra root solve.
    On hard MILPs that one-time cost is dwarfed by the node reductions, but on
    trivial instances it dominates. GMI is therefore **opt-in, off by default**
    (`discopt.solver.GOMORY_CUTS_ENABLED`); when off, the call site passes no
    integer indices so the cut loop runs exactly as it did before GMI (no
    added per-solve cost). Two pure optimizations apply when it is on: skip the
    basis-recovery/GMI work unless an integer variable is fractional at the
    vertex, and stop after round 0 when there are no cover/clique cuts to
    re-separate. `TestGomoryWiring` enables the flag explicitly.
    *(The interim size-gate, `GOMORY_MIN_INT_VARS`, was superseded by Path B
    below: it kept the suite green but still ~15 min, because the suite's larger
    JAX-mode instances tripped the threshold and recompiled. The real fix was to
    remove the recompile, not to tune a threshold around it.)*
  - **Increment 6 — Path B: POUNCE as the primary node engine (the real fix).**
    The recompile is a property of *shape-varying JAX*: the MILP B&B solved node
    relaxations with a `vmap`'d JAX IPM, so cut-augmented shapes recompiled. In
    POUNCE-only mode, node relaxations are now solved by **POUNCE (Rust)**
    directly (`_solve_node_lp_pounce`) on the *augmented* standard form — any
    shape, no recompile, and the `OPTIMAL` bound is KKT-valid (no trust-gate
    decertification; the rare stall/unavailable node still defers to the
    existing POUNCE recovery / gap-decertification). POUNCE was already wired in
    for *recovery*; this promotes it to the primary engine. Consequently GMI is
    re-gated as a **POUNCE-mode feature**: `_gomory_enabled(prefer_pounce)` turns
    it on exactly when node solves are POUNCE (cut shapes free), and off under
    the JAX IPM (where cuts would recompile). Net result: the JAX default path
    is back to baseline (GMI off, no recompile), and POUNCE mode gets the cut
    benefit cheaply — node reductions with no per-shape recompile. Verified: the
    `test_milp_pounce` HiGHS battery (POUNCE mode, GMI now auto-on) passes;
    `TestGomoryGate` covers the engine gate + a POUNCE-mode correctness check on
    the formerly-failing knapsack. This trades away batched-GPU node waves in
    POUNCE mode (POUNCE solves sequentially), an accepted call given GPU is not
    a near-term priority.
  - **Increment 6b — cut-loop root solve through POUNCE — DONE.**
    `_cut_loop_relaxation_x` routes the cut loop's own root relaxation solve
    through POUNCE in POUNCE mode (JAX IPM otherwise), so the residual recompile
    when GMI augments the shape mid-loop is gone too. The relaxation point is
    only a separation seed, so cut validity is unaffected by the engine.
  - **Increment 7 — MIR cuts from original rows — DONE.** `lp/mir.rs`
    (`separate_mir`) separates Mixed-Integer Rounding cuts directly from the
    model's `≤` rows, complementing GMI (which applies the MIR function to
    *tableau* rows): for `Σ a_j x_j ≤ b` shifted to `x' = x − l ≥ 0`, with
    `f = frac(b')`, integer columns get `⌊a_j⌋ + (f_j − f)^+/(1−f)` and
    continuous columns `min(a_j,0)/(1−f)`, rhs `⌊b'⌋`. Single-row MIR is only as
    strong as the row scaling, so each row is tried at `δ = 1` and `1/|a_j|` per
    integer column and the most-violated valid cut is kept, then mapped back to
    the original `x`. Basis-free and structural (O(1) coefficients), so it
    composes with Path B without recompile; numerically guarded
    (`FRAC_MIN`/`MAX_ABS_COEFF`/dynamism) and validated by exhaustive
    integer-hull enumeration over random rows (`lp::mir` tests). Exposed via
    `discopt._rust.mir_cuts_py`, wired into the cut loop alongside GMI under the
    same POUNCE-mode gate, and added with its own `≤`-slack
    (`_augment_lpdata_with_mir_cuts`). `TestMirCuts` covers the binding and a
    POUNCE-mode end-to-end correctness check against HiGHS. Upper-bound
    complementation and aggregation-based c-MIR (multi-row) are the strengthening
    follow-ups.
  - **Open (future):** c-MIR aggregation + upper-bound complementation; GMI/MIR
    re-separation across rounds; and, if batched-GPU node solving becomes a
    priority, a fixed-shape (padded / shape-polymorphic) JAX IPM so cuts stay
    cheap there too.
- **Basis cuts:** Gomory mixed-integer and MIR at the root and at periodic
  re-solves, feeding the existing `CutPool`
  (`python/discopt/_jax/cutting_planes.py`; cap/aging/dedup already there).
  Cuts are added to the formulation and the tree resumes batch IPM solving
  — explicitly **not** a per-node cut-and-resolve loop.
- Side benefit: the recovered vertex yields clean fractional branching
  candidates, improving branching independent of cuts.

### Phase 3 — Cut & heuristic suite (compete II)

- **Basis-free cuts that batch well:** knapsack cover, clique cuts (finally
  consuming the pairwise clique table from `presolve/cliques.rs`, which
  currently dead-ends), flow cover, implied bounds.
- **Primal heuristics** designed to exploit wide batches — batched diving,
  RINS, local branching, feasibility-pump polish. Batched
  diving/strong-branching waves are discopt's asymmetric advantage over a
  sequential simplex solver.
- **Conflict analysis:** learn constraints from infeasible nodes (today
  they are silently fathomed), seeded by the Farkas certificates from P0.2.

### Phase 4 — Retire remaining HiGHS consumers

- OA / GDP-LOA masters (`oa.py`, `gdpopt_loa.py`) → the self-hosted integer
  solver. This closes the convex-MINLP loop (single-tree LP/NLP-BB becomes
  natural here) without HiGHS.
- `_jax/obbt.py` → POUNCE LP. McCormick-LP mode (`mccormick_lp.py`,
  `milp_relaxation.py`), `partition_selection.py`, DOE optimal design, RO
  polyhedral subproblems, `export`/`cli` incidentals.
- **Packaging — already done; no removal needed.** `highspy` was never a
  hard runtime dependency: `pyproject.toml` lists only `jax`/`jaxlib`/`numpy`/
  `scipy` under `dependencies`, with HiGHS behind the `highs` extra (and the
  `dev` extra) and POUNCE behind the `pounce` extra. The thing that was
  actually missing was code paths that *hard-required* HiGHS even in
  POUNCE-only mode; once those go through the selector (above),
  `pip install discopt[pounce]` is fully functional with no HiGHS present.
  **We deliberately keep `highspy` selectable.** discopt is likely not yet
  performance-competitive with HiGHS, and HiGHS remains valuable as (a) the
  default, faster engine when installed (the selector is HiGHS-first unless
  `nlp_solver="pounce"`) and (b) the reference oracle for correctness/basis
  tests. So the end state is *HiGHS-optional at runtime*, not HiGHS-free.

### Phase 5 — Parity push + differentiability completion

- Benchmark-driven gap closing vs HiGHS (MIPLIB subset) and BARON
  (MINLPLib), tuning cuts/branching/heuristics against the existing phase
  gates in `discopt_benchmarks/config/benchmarks.toml`.
- **Differentiable MILP/MIQP as a first-class feature:** extend the
  custom_jvp / KKT implicit-differentiation machinery (T22/T23) to wrap the
  integer solve — differentiate through the final relaxation and active set
  at the incumbent. This is the headline capability the POUNCE-only bet
  enables; it ships with tests and a notebook, not as a side effect.
  (CPU-sufficient: the implicit-diff step is a small dense KKT solve.)

---

## 7. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| Pure-MILP performance trails HiGHS for a long time | High | Lead with differentiability/batch as the value proposition; gate parity on benchmarks, never promise a date |
| Crossover numerical robustness | High | Dedicated sub-project; HiGHS basis as test oracle; fallback to basis-free cut families |
| Correctness regressions during migration | High | HiGHS retained as CI oracle through Phase 4; `incorrect_count = 0` gates every phase |
| IPM warm-start weakness inflates per-node cost | Medium | Batch width as the throughput lever; root/periodic cuts, never per-node loops |
| Smeared interior solutions degrade branching | Medium | Purification (Phase 1), crossover (Phase 2) |
| POUNCE feature/maturity gaps surface late | Medium | Phase 0 is the forcing function; keep cyipopt fallback until the Phase 1 gate passes |
| POUNCE wheel-building across platforms | Medium | CI wheels via maturin from Phase 0 — packaging *is* the promise |

---

## 8. Open questions

1. Does POUNCE expose any interior warm-start surface (initial dual/slack
   point beyond `x0`)? The wrapper currently passes only `x0`. Low priority
   given §4's "batch width over warm starts" policy, but worth confirming
   in the POUNCE crate.
2. Crossover scope: full primal-dual crossover vs primal-only vertex
   recovery first? Recommend primal-only as the Phase 2 MVP, dual crossover
   only if MIR separation needs it.
3. Should the McCormick-LP mode (`mccormick_lp.py`) migrate to POUNCE LP
   (Phase 4) or be subsumed by the self-hosted MILP path once cuts land?
   Defer until Phase 3 data exists.
