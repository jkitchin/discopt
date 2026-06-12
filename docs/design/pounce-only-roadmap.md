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

- **P0.1 Pure-LP path.** Validate POUNCE on zero-Hessian / pure-linear
  problems: degeneracy, dual degeneracy, unboundedness, free variables,
  empty constraint sets. LP is the inner loop of everything downstream
  (OBBT, McCormick-LP, masters). Add an LP-specific test matrix mirroring
  `test_lp_highs.py`.
- **P0.2 Infeasible-node handling.** The cyipopt status map inherited by
  `nlp_pounce.py` sends Ipopt status 2 (*locally* infeasible) to `ERROR`.
  For an IPM this means "this start found no feasible interior point," not
  "the node is infeasible" — pruning on it is unsound. Required behavior:
  feasibility-restoration retry / restart on local infeasibility; declare a
  node infeasible **only** on a genuine Farkas dual ray (infeasibility
  certificate). Expose the certificate through `NLPResult`.
- **P0.3 Bound trust-gate.** Use a POUNCE objective as a valid dual bound,
  and its multipliers for reduced-cost fixing, **only** when the solve
  converged to tolerance. An unconverged IPM bound used for fathoming can
  cut the optimum — this violates the `incorrect_count ≤ 0` invariant.
- **P0.4 Batch LP/QP.** Extend the `solve_nlp_batch` path to LP/QP node
  waves; extend `nlp_backend.py` so the seam dispatches LP/QP as well as
  NLP, keeping call sites engine-agnostic.
- **P0.5 HiGHS as CI oracle.** From this phase on, cross-check every POUNCE
  LP/QP result against HiGHS in CI (test-only dependency). HiGHS stops
  being a runtime engine long before it stops being a correctness guard.

### Phase 1 — Self-hosted integer B&B ("cover" milestone)

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

- **Pure-Rust crossover** (interior → basic feasible vertex + basis).
  Keystone deliverable; budget as its own sub-project with heavy numerical
  testing. Use HiGHS basis output as the test oracle.
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
- Remove `highspy` from runtime dependencies in `pyproject.toml`; keep it
  in the dev/test extra as the CI oracle. **discopt is now HiGHS-free at
  runtime.**

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
