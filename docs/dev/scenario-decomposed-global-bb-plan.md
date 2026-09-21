# Scenario-decomposed global branch and bound (A8)

**Date:** 2026-09-21
**Status:** entry experiment run; see §5 for the verdict
**Audience:** an implementing agent executing this plan phase by phase.
**Scope:** a new global method for two-stage stochastic nonconvex NLPs, living in
`python/discopt/stochastic/`, reusing `python/discopt/decomposition/` (structure,
parallel) and `python/discopt/_relax/` (relaxation, FBBT/OBBT).
**Relationship to existing docs:**

- `docs/dev/decomposition-remediation-plan.md` §1 registry entry **A8** — "Nonconvex
  models: GBD runs heuristically (`bound=None`). No NGBD-style rigorous alternative."
  This plan is a *second* answer to A8, structurally different from the NGBD spike
  (T3.3, recorded "not run" in that plan's decision log).
- `docs/dev/stochastic-module-plan.md` — the module this lands in. Today's methods
  (`solve_lshaped`, `progressive_hedging`) are convex-exact / heuristic on nonconvex
  models; this is the globally-valid sibling.
- `docs/design/relaxation-catalog.md` — the relaxation layer this method *uses*; it does
  not replace it (see §1.3).

---

## 0. Provenance, and two assumptions that were wrong

The method is Cao & Zavala (2019) — `CaoZavala2019` in `docs/references.bib`,
implemented by the authors as **SNGO** (Julia). **SNoGloDe** (`Stinchfield2025SNoGloDe`)
is the Python/Pyomo successor from the Stinchfield/Laird group, which generalises the
same scheme from two-stage stochastic programs to general block-angular structures; no
public repository or DOI for it was locatable as of 2026-09-21 (it appears as an
optional `import snoglode` inside `sandialabs/sparow`).

This plan was first drafted from the abstract. Reading the full preprint **falsified two
premises of that draft**, both recorded here per CLAUDE.md §4/§11:

1. **"The method needs a small first stage (single digits)." FALSE.** §5.2 of the paper
   solves a parameter-estimation problem with a **48-variable first stage** (temporal
   decomposition: two parameters plus 46 linking states) in 8 min–2 h across 12 real
   datasets, where SCIP exceeds 12 h on 9 of them.
2. **"The non-anticipativity relaxation is the λ=0 Lagrangian bound, so it is weak."**
   Weak *relative to a Lagrangian dual*, yes; but measured against a spatial solver's
   root bound it is not. The paper's root gaps (which are exactly the EVPI) on four
   profiled instances are **36.1 / 14.3 / 1.4 / 9.5 %**, versus SCIP's root gaps of
   **101.5 / 62.6 / ≥10000 / 10.6 %**. The relevant baseline is the monolithic
   relaxation, not the best achievable dual bound.

Both corrections widen the case for building it. §4 exists because neither number was
measured against *discopt's* root bound.

---

## 1. The algorithm

### 1.1 Problem and bounds

For `z = min_{x ∈ X0} Σ_s Q_s(x)` with `Q_s(x) = min_{y_s} { f_s(x,y_s) : g_s(x,y_s) ≤ 0 }`
(probability weights folded into `f_s`), at a node with first-stage box `X ⊆ X0`:

- **Lower bound** (paper §2.1) — lift by replicating `x` per scenario and *drop* the
  non-anticipativity constraints `x_s = x_{s+1}`:

      β(X) = Σ_s β_s(X),   β_s(X) = min_{x_s ∈ X, y_s} f_s(x_s, y_s)

  `β(X) ≤ z(X)` because the lifted feasible set contains the original, and `β` is
  monotone: `β(X₁) ≥ β(X₂)` for `X₁ ⊂ X₂`. If any `β_s(X) = ∞` the node is infeasible.
  **Each `β_s` must come from a global solve** — this is the correctness hinge (§3).
- **Upper bound** (paper §2.2) — fix `x̂ ∈ X` and evaluate `α(X) = Σ_s Q_s(x̂)`, again S
  independent global solves.
- **Branching** is on the first-stage variables only; second-stage branching happens
  inside the subproblem solves.

The root gap `α(X0) − β(X0)` is the **expected value of perfect information**. The whole
method is a bet that EVPI is moderate on the target class.

### 1.2 Convergence (paper §3)

Proved by adapting Horst & Tuy (`HorstTuy1996`, already in `references.bib`):

- *Exhaustive subdivision* (branch the longest edge, keep branch points away from the
  bounds) ⇒ "delete by infeasibility" is certain in the limit (Lemma 2) and the lower
  bounding operation is **strongly consistent** (Lemma 3). This half needs only lower
  semicontinuity of `Q_s`, which follows from compactness of `Y_s(x)` (Assumption 2).
- *Upper bound convergence* (Lemma 6) needs **Assumption 3: `Q(·)` Lipschitz continuous
  near `x*`**, which holds if MFCQ holds at the scenario subproblems. This is the one
  constraint qualification in the proof, and it is on the UB half only.
- Bound-improving node selection (always split a node attaining the incumbent lower
  bound) then gives `α_k → z`, `β_k → z` (Theorem 1).

### 1.3 What the reference implementation does *besides* the two bounds (paper §4)

These are not optional extras; they are where SNGO's performance comes from, and four of
the five map onto machinery discopt already has.

| SNGO component | discopt counterpart |
|---|---|
| LP relaxation of the **whole** stochastic problem (convexification + OA, αBB and RLT cuts); node bound = `max(LP bound, β(X))` | `_relax/` relaxation compiler, αBB, RLT — **already ours**; the decomposed bound is a *co-bound*, not a replacement |
| Subproblems warm-started from the LP relaxation solution | `warm_start.py` / `initial_solution=` |
| FBBT + **OBBT at every node** (2·n_x LPs; affordable because subproblem solves dominate) | `tightening.py`, FBBT in `crates/discopt-core` |
| Strong branching on first-stage vars scored by LP-relaxation improvement; fallback to widest range; branch point `Σ x̃_s/|S|`; skip vars with range < 1e-4 | new (small) |
| UB every node for the first 3 levels then every 2 levels | new (trivial) |

Two bound-strengthening tricks worth copying verbatim:

1. **Inherited scenario cut** `f_s(x,y_s) ≥ β_s(X)`, valid for every descendant by the
   monotonicity above. SNGO keeps an auxiliary objective variable per scenario and
   tightens its lower bound down the tree.
2. **Subproblem reuse**: if the parent's minimiser `x̃_s` lies in a child's box, the
   child's `β_s` equals the parent's — no re-solve. Given that **>90 % of runtime is the
   scenario solves** (paper §4), this is the single largest practical win available.

### 1.4 Measured cost model (paper §4, §5.1)

Per-node cost grows **linearly in S**; node count does **not** grow with S, because the
branching dimension is `n_x`. On the PID instance SNGO visited **22,410× fewer nodes**
than SCIP at **1,102× the cost per node**, and won by an order of magnitude in wall time.
The reference implementation is **fully serial**; the authors list parallelisation as
future work and call it "challenging because of memory management and load imbalancing".

---

## 2. Why this fits here

- **The structure layer already produces the inputs.** `decomposition/structure.py`'s
  `DecompositionStructure` carries `blocks`, `complicating_vars`, `coupling_constraints`,
  and `restricted_bounds`/`flat_bounds` give the node box.
- **The parallel layer already exists and is what SNGO lacked.**
  `decomposition/parallel/comm.py` maps per-block work with a *fixed reduce order*, so
  `backend="threads"` returns bit-identical bounds. Steps 1, 2, 5, 6 of the paper's cost
  breakdown are all embarrassingly parallel.
- **The `Model.subset` blocker does not apply to the stochastic entry point.**
  `stochastic/scenario.py:8` states it directly: the scenario-creator callback pattern
  means "no expression-graph surgery is needed". A standalone per-scenario `Model` is
  built by calling the user's `recourse_builder` against a *fresh* `Model` — which is why
  the general block-angular case (SNoGloDe's generalisation) stays out of scope (§7).
- **The one API gap** is that today's builder signature closes over the outer model's
  first-stage variables (`python/tests/test_stochastic_phase1.py:32-39` — the inner `recourse`
  closure captures `q`, a variable of the *outer* `m`), so it cannot be retargeted. See T1.1.

---

## 3. Correctness contract (binding)

1. **`β_s` is a dual bound, never an incumbent.** The node lower bound sums
   `SolveResult.bound` over scenarios, and only when `bound_valid` is true. Summing
   `objective` (or a local NLP value) yields a "lower bound" *above* the true optimum,
   which this solver would then certify — the exact false-`optimal` failure CLAUDE.md §1
   forbids. If any scenario returns no valid bound, the node bound is `-inf`.
2. **A scenario solve that hits a limit still contributes its dual bound**, not its gap
   midpoint, and the node must be marked non-certifying if any subproblem is uncertified.
3. **`α` is only an upper bound once the point is feasible.** Fixed-`x̂` scenario
   solutions go through the existing incumbent verification (`verify_incumbent`) before
   updating the global incumbent.
4. **Exhaustiveness is a correctness property, not a heuristic.** The branch point must
   keep a minimum distance from the box bounds, and the "skip variables with range
   < 1e-4" rule must not be allowed to stall subdivision on a variable that still carries
   gap — if every candidate is below the threshold the node is closed as ε-optimal, not
   silently dropped.
5. **Assumption 3 (MFCQ / Lipschitz `Q`) is an assumption of the UB half of the
   convergence proof.** It cannot be checked cheaply. It does not threaten validity of
   any bound we report — an unverified `α` is still a real feasible point's value — only
   the finite-termination claim. Say so in the docstring rather than asserting
   convergence unconditionally.

---

## 4. Entry experiment (run before any implementation)

Per CLAUDE.md §4, the falsifying experiment runs first.

- **Script:** `discopt_benchmarks/scripts/stochastic_evpi_entry_experiment.py`
- **Hypothesis:** on two-stage stochastic nonconvex NLPs, the decomposed root bound
  `β(X0)` is tighter than discopt's monolithic root bound (`SolveResult.root_bound`), and
  the advantage does not decay as S grows.
- **Measurement:** per (family, S) cell — `LB_dec = Σ_s β_s` from S global subproblem
  solves; `α` from the mean-candidate fixed-`x̂` re-solve; discopt's `root_bound` on the
  extensive form, replicated (the baseline is given its *best* root bound across
  replicates, so the comparison is generous to it); both gaps taken against the best
  incumbent either arm found.
- **Families:** PID controller tuning (paper §5.1, `n_x = 3`), temporal-decomposition
  parameter estimation (paper §5.2, `n_x = 2 + (S−1)`), and a two-stage pooling/blending
  problem with a bilinear pool-quality equality (`n_x = 2`) — the nonconvexity class the
  in-repo corpus is actually full of.
- **Known limitation, stated up front:** there are **no two-stage stochastic instances in
  the in-repo corpus** (`python/tests/data/minlplib_nl/`) or in MINLPLib, so this
  experiment cannot be run on real corpus instances the way CLAUDE.md's working-on-an-
  issue §2 prefers. The families are the reference paper's own problem classes plus one
  drawn from our corpus's dominant nonconvexity. The residual risk — that the result does
  not transfer to the instances a user would bring — is real and is the reason the
  implementation phases below stay staged behind further measurement.
- **Kill criterion:** if `decomposed_root_gap ≥ discopt_root_gap` in a majority of cells,
  the method has no bound headroom here and the implementation is **not** scheduled.

Run it with:

```bash
python -u -m discopt_benchmarks.scripts.stochastic_evpi_entry_experiment \
    --families pid,estimation,pooling --scenarios 4,8,16 \
    --time-limit 120 --sub-time-limit 30 --replicates 3 \
    --out reports/stochastic_evpi_entry.json
```

---

## 5. Results

<!-- RESULTS -->

---

## 6. Implementation phases (conditional on §5)

Each task names files, algorithm, tests and an exit gate. Phases run in order.

### Phase 0 — make a scenario buildable standalone

**T0.1 — retargetable recourse builder.** Extend the builder contract to
`recourse_builder(model, data, s, first_stage)` (keyword-optional, so today's
3-argument callables keep working — inspect the signature, do not guess), and add
`first_stage_builder(model) -> dict[str, Variable]` so `build_extensive_form` and a new
`build_scenario_model(s)` construct the *same* first stage in different `Model`s.
*Files:* `python/discopt/stochastic/extensive_form.py`, `scenario.py`.
*Gate:* a test builds the monolith and the S standalone subproblems from one pair of
callbacks and asserts that fixing the first stage at a point gives
`Σ_s p_s · sub_s = monolith` to 1e-9 — with an **executed-comparison count** asserted
non-zero (CLAUDE.md measurement §6).

### Phase 1 — the bounds, no tree

**T1.1 — `root_bounds(...)`**: compute `β(X0)`, `α(X0)` and the EVPI for a declared
two-stage model, through `decomposition/parallel/comm.py` so the reduce order is fixed.
*Gate:* on the §4 families, `β ≤ z ≤ α` holds on every instance; thread and sequential
backends agree bit-for-bit; ≥1 comparison asserted per instance.

**T1.2 — bound hygiene.** `β_s` sums `bound` only when `bound_valid`; a limit-terminated
subproblem degrades the node, never inflates it. *Gate:* a regression test that forces a
1 ms subproblem budget and asserts the node bound goes to `-inf` rather than to a sum of
incumbents.

### Phase 2 — the tree

**T2.1 — first-stage-only spatial B&B** in Python (per-node cost is S global solves, so
loop overhead is noise; the Rust tree is not touched). Node selection = best bound;
subdivision = exhaustive (longest edge, branch point `Σ x̃_s/|S|` clamped away from the
bounds); delete-by-infeasibility when any `β_s = ∞`.
*Gate:* on each §4 family the returned `objective`/`bound` bracket the extensive-form
solve's certified optimum, and `gap_certified` is never set when any subproblem was
uncertified.

**T2.2 — inherited scenario cut + subproblem reuse** (§1.3). *Gate:* bound-neutral —
node count and certified objective **exactly unchanged**, wall time down; per CLAUDE.md
§5's bound-neutral regime any drift means the change is wrong.

**T2.3 — co-bound with the existing relaxation**: node bound = `max(relaxation bound,
β(X))`, warm-starting subproblems from the relaxation solution. *Gate:* differential —
node counts non-increasing on the panel, no bound above a known optimum.

### Phase 3 — the accelerators

**T3.1** OBBT at every node over the first stage (2·n_x LPs).
**T3.2** Strong branching scored by relaxation-bound improvement, widest-range fallback.
**T3.3** UB schedule (first 3 levels, then every 2).
**T3.4** `backend="threads"` for the per-node subproblem map.
*Gate for the phase:* wall-clock improvement measured with an interleaved control and a
reported spread (CLAUDE.md measurement §9); bound-neutrality checked for T3.1–T3.3.

### Phase 4 — surface

`discopt.stochastic.solve_global(...)` mirroring `solve_lshaped`'s signature; a
`MethodKind` member for the advisor (`advisor/types.py` registers `DANTZIG_WOLFE`,
`ADMM`, `SCHUR`, … but nothing for a globally-valid decomposition); `Soundness` for it is
`PROVEN_EQUIVALENT` on nonconvex models, which no current candidate can claim.

---

## 7. Out of scope (and why)

- **Integer first-stage variables.** The paper's own future work; branching on integers
  interacts with the exhaustiveness argument and needs its own treatment. `solve_lshaped`
  and `Laporte1993`'s integer L-shaped cuts remain the route for those.
- **Risk measures.** CVaR couples scenarios through the objective and destroys the
  decomposition — the same reason `stochastic/lshaped.py` ships risk-neutral first.
- **General block-angular structures (the SNoGloDe generalisation).** Requires extracting
  a standalone sub-`Model` from a monolith, i.e. the `Model.subset(vars, constraints)`
  primitive whose absence deferred T1.4 in the decomposition remediation plan. The
  stochastic entry point sidesteps it via the builder callbacks; the general case does
  not. Revisit once that primitive exists.
- **MPI.** The thread backend is the shipping target; `comm.py` reserves the slot.

---

## 8. Decision log (append per entry)

- **2026-09-21 — plan drafted from the abstract, then corrected against the full
  preprint.** Two premises falsified; see §0. Neither error would have been caught
  without reading the paper, which is the §4 entry-experiment discipline applied to the
  literature rather than to code.

<!-- DECISION LOG -->

---

## 9. References

- Cao, Zavala (2019). *A scalable global optimization algorithm for stochastic nonlinear
  programs.* J. Global Optim. 75(2). `CaoZavala2019`. Implemented as SNGO (Julia).
- Stinchfield, Bhatia, Bynum, Cao, Laird (2025). *SNoGloDe: a structured nonlinear global
  decomposition framework.* `Stinchfield2025SNoGloDe`. Generalises the above to
  block-angular structures; Python/Pyomo.
- Li, Tomasgard, Barton (2011). *Nonconvex generalized Benders decomposition for
  stochastic separable MINLPs.* JOTA 151. `LiTomasgardBarton2011`. The other answer to
  A8 (the T3.3 spike in the decomposition remediation plan).
- Horst, Tuy (1996). *Global optimization: deterministic approaches.* `HorstTuy1996`.
  Supplies the convergence machinery §1.2 adapts.
- Laporte, Louveaux (1993). *The integer L-shaped method…* `Laporte1993`.
