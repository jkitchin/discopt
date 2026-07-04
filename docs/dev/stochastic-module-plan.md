# Stochastic Programming Module — Design & Implementation Plan

**Date:** 2026-07-03
**Status:** design plan for a new `python/discopt/stochastic/` module.
**Thesis:** discopt has **robust** optimization (worst-case over an uncertainty set)
but no **stochastic** optimization (expected / risk-adjusted cost over a probability
distribution with recourse). The decomposition engine already shipped — multicut
Benders (= the **L-shaped method**), GBD (convex nonlinear recourse), and Lagrangian
dual decomposition (the basis for **progressive hedging**) — *is* the stochastic-
programming solution machinery. So SP is a **scenario/expectation front-end + one
missing engine primitive (probability weighting) + a risk-measure layer**, not a new
solver. It is the distributional sibling of the `ro/` module.

---

## 0. What SP is, and why discopt is close

A two-stage stochastic program is

```
  min_x   c·x + E_ξ[ Q(x, ξ) ]            first-stage (here-and-now) decision x
  s.t.    A x <= b
  where   Q(x, ξ) = min_y { q(ξ)·y : W(ξ) y <= h(ξ) − T(ξ) x }   recourse (wait-and-see)
```

The first stage `x` is chosen before the uncertainty `ξ` is realized; the recourse
`y` adapts to each realization. With a finite scenario set `{ξ_s, p_s}` (given, or
sampled via **SAA**), the expectation is `Σ_s p_s Q(x, ξ_s)`.

**The reduction discopt is one step from:** each scenario's recourse `Q(x, ξ_s)` is a
*block* that couples to the first stage only through `x`. That is exactly the
two-stage structure the decomposition engine already exploits:

| SP concept | discopt asset | Status |
|---|---|---|
| first stage `x` / recourse `y_s` | `Model.first_stage(...)` / `second_stage(...)` annotations → `DecompositionStructure.complicating_vars` | exists |
| per-scenario recourse block | multicut Benders per-block `η_b` + `_add_opt_cut` (`benders/solver.py`) | exists |
| convex nonlinear recourse | `solve_gbd` + its convexity gate | exists |
| dual decomposition / PH substrate | `solve_lagrangian` (subgradient/bundle) + reserved `MethodKind.PROGRESSIVE_HEDGING` | engine exists, PH driver to build |
| scenario-parallel subproblem solves | `parallel.comm.select_backend`, `DecomposedModel.map_subproblems` (deterministic, input-order) | exists |
| soundness ledger | `ir/certificate.py` `SoundnessCertificate`, `Soundness` enum (`PROVEN_EQUIVALENT` / `RELAXATION` / `HEURISTIC`), `assert_sound()` | exists |
| **scenario probabilities `p_s`** | — the engine sums block contributions with implicit **weight 1.0** (`solver.py` `total += res[1]`; Lagrangian `sub_total += sub_lb`) | **must add** |
| expectation / CVaR / chance objectives | — | **must build** (the SP front-end) |

The single missing *engine* primitive is **probability weighting**: nothing in the
current cut/bound summation accounts for `p_s`. Everything else is a front-end that
builds scenario-expanded models and hands them to the existing drivers.

---

## 1. Scope (v1) and explicit non-goals

**In scope (v1):**
- **Two-stage** SP with **linear or convex** recourse.
- **Scenario sources:** an explicit `{(ξ_s, p_s)}` list, or **SAA** — sample a
  distribution to a finite scenario set (with the sample count logged; no silent
  truncation).
- **Two solution methods**, sharing one front-end:
  1. **Extensive form (deterministic equivalent)** — build the single monolithic
     model `min c·x + Σ_s p_s q_s·y_s` and solve it directly. Correct by
     construction (`Soundness.PROVEN_EQUIVALENT`); the baseline and the **oracle**
     for everything else. Scales poorly (all scenarios in one model).
  2. **L-shaped (probability-weighted multicut Benders)** — the scalable path;
     rigorous bound for convex recourse via the existing Benders/GBD gate.
- **Risk measures as first-class constructs:** `expectation`, **CVaR**
  (Rockafellar–Uryasev), and **chance constraints** (SAA + indicator/GDP).
- **Mixed-integer *first* stage** is fine (integrality on `x` does not break the
  scenario decomposition; solved globally through the MINLP master).

**Out of scope in v1 — refused loudly, not silently approximated (CLAUDE.md §3):**
- **Integer *recourse*** — the recourse value function `Q(x, ξ)` is then nonconvex/
  discontinuous, so L-shaped optimality cuts are invalid. v1 routes integer-recourse
  models to the **extensive-form MINLP** (still exact, just not decomposed) with a
  logged note, and refuses the L-shaped path for them. (Integer L-shaped / Laporte–
  Louveaux is a Phase 3 item.)
- **Multistage** (>2 stages, scenario *trees*) — the `NESTED_BENDERS` / SDDP slot is
  reserved (`advisor/types.py`) but a full nested driver is Phase 3.
- **Distributionally-robust (DRO)** — worst-case *over a set of distributions* is an
  RO×SP hybrid; a later bridge to `ro/`, not v1.

---

## 2. Module layout

```
python/discopt/stochastic/
  __init__.py            # TwoStageProblem, ScenarioSet, expectation, cvar, chance_constraint
  scenario.py            # Scenario, ScenarioSet; SAA sampling from distributions
  problem.py             # TwoStageProblem front-end + .formulate()/.solve()
  risk.py                # expectation / cvar / chance_constraint objective & constraint builders
  extensive_form.py      # deterministic-equivalent builder (baseline + oracle)
  lshaped.py             # probability-weighted L-shaped driver (wraps solve_benders)
  ph.py                  # progressive hedging (Phase 2)
```

Mirrors `ro/`'s flat, builder-pattern shape. No new external deps (decomposition and
`_jax` are core).

---

## 3. Key design pieces

### 3.1 Scenario representation (`scenario.py`)
A `Scenario` binds a probability `p_s` to a *realization* of the uncertain
quantities — which are discopt `Parameter`s whose value varies by scenario:

```python
ScenarioSet.from_list([(0.3, {demand: 100, price: 9}), (0.7, {demand: 150, price: 8})])
ScenarioSet.sample(distributions={demand: Normal(120, 20)}, n=200, seed=...)   # SAA
```
Probabilities are validated to sum to 1 (loud error otherwise). SAA records `n` and
the seed on the set (reproducibility; no hidden sampling). Uncertain quantities must
be declared `Parameter`s so the scenario values substitute cleanly (the same
`substitute_param` idiom `ro/` uses).

### 3.2 Probability weighting — the one new engine primitive
Two integration paths, deliberately ordered easy→scalable:

- **Extensive form (no engine change):** the weighting lives *in the model* —
  `minimize c·x + Σ_s p_s (q_s·y_s)`. The existing solver handles it verbatim.
  Correct by construction; this is why it is the v1 baseline and oracle.
- **L-shaped (the primitive to add):** the Benders master is `min c·x + Σ_s η_s`
  with per-scenario cuts. To make `Σ_s η_s` equal the *expectation*, scale each
  scenario subproblem's objective by `p_s` before the cut is generated — then
  `_add_opt_cut`'s dual-based cut (`η_s >= (p_s q_s)·…`) already sums to
  `Σ_s p_s Q_s`. This is a **local, provably valid** change: scaling a subproblem's
  objective by `p_s > 0` scales its optimal dual by `p_s`, so the complete-dual cut
  (`benders/solver.py:_dual_const`/`_add_opt_cut`) remains valid, just weighted.
  Implemented either by pre-scaling the recourse cost vector per block, or by adding
  a `block_weights` argument threaded into the multicut aggregation. **Acceptance:
  the weighted cuts must reproduce `Σ_s p_s Q_s(x)` to machine tolerance vs the
  extensive-form objective** (differential test).

### 3.3 Risk measures (`risk.py`)
- **Expectation** `E[Q] = Σ_s p_s Q_s(x)` — the default objective.
- **CVaR_α** via Rockafellar–Uryasev:
  `min η + 1/(1−α) Σ_s p_s [Q_s(x) − η]^+`, introducing an auxiliary `η` and per-
  scenario excess variables `u_s >= Q_s − η, u_s >= 0`. Linear when recourse is
  linear; keeps the problem in the certified path. A `mean_cvar(λ)` convex
  combination for risk-averse trade-offs.
- **Chance constraint** `P(g(x, ξ) <= 0) >= 1 − ε`: under SAA, an indicator per
  scenario (`z_s ∈ {0,1}`, `g(x, ξ_s) <= M z_s`, `Σ_s p_s z_s <= ε`) — reuses the
  existing `if_then`/GDP machinery. Documented as the SAA approximation (finite-
  sample), with the sample count surfaced.

### 3.4 User-facing API (`problem.py`) — mirrors `RobustCounterpart`
```python
from discopt.stochastic import TwoStageProblem, ScenarioSet, cvar

sp = TwoStageProblem(
    model,
    first_stage=[x],
    recourse=[y],
    scenarios=ScenarioSet.from_list([...]),   # or .sample(...)
    recourse_objective=q_expr,                 # per-scenario second-stage cost
    recourse_constraints=[...],
)
sp.set_objective(cvar(alpha=0.95))             # or expectation() (default)
result = sp.solve(method="lshaped")            # or "extensive_form"
```
`solve()` (guarded like `RobustCounterpart.formulate()`): validates the scenario set,
gates recourse convexity for `lshaped` (reuse the certifier), builds the chosen
form, and dispatches — extensive form → `model.solve()`; L-shaped → `solve_benders`
with a per-scenario `DecompositionStructure` (`blocks = scenarios`,
`complicating_vars = first_stage`) plus the probability weights. Records the
appropriate `SoundnessCertificate` (`PROVEN_EQUIVALENT` for extensive form,
`RELAXATION`→converged for L-shaped).

---

## 4. Soundness (the non-negotiables)

1. **Extensive form is the ground truth.** It is an exact deterministic equivalent
   (`PROVEN_EQUIVALENT`); every other method is validated to reproduce its certified
   optimum. This is the SP analog of "the measurement wins."
2. **L-shaped bound validity requires convex recourse** — reuse the Benders/GBD
   convexity gate (PR #421 confirmed it withholds `bound`/`gap_certified` on
   nonconvex recourse). Integer recourse is refused on the L-shaped path.
3. **Probability weighting is verified, not assumed** — the weighted cuts/bound must
   equal `Σ_s p_s Q_s` before any SP result is trusted (the primitive is fuzzed like
   `symbolic_diff` in the bilevel plan).
4. **Risk-measure formulas are fuzz-verified** against a brute-force
   sort-and-average CVaR / empirical chance probability on small scenario sets.
5. **SAA is honest** — the scenario count and seed are recorded and surfaced; a
   chance constraint is labeled a *finite-sample* approximation, never presented as
   the true probabilistic guarantee. (SAA confidence intervals are a Phase 3 add.)

---

## 5. Phased plan

| Phase | Deliverable | Acceptance | Docs |
|---|---|---|---|
| **0 ✅ DONE** | `scenario.py` (`ScenarioSet` / SAA), `risk.py` (`Expectation`/`CVaR`/`MeanCVaR`/`chance_constraint`), `extensive_form.py` (deterministic equivalent via a scenario-creator callback) | the extensive-form objective carries the scenario probabilities (`Σ p_s Q_s`, verified by compiling+evaluating it); the **CVaR RU expression equals the analytic CVaR** (brute-force η-grid oracle, α∈{0.5,0.8,0.95}); scenario-set validation + reproducible SAA; chance-constraint coverage encoding — **9 tests pass, ruff clean**. Certified end-to-end newsvendor/farmer solves → CI (need the solver). | module/API docstrings; notebook seeded in Phase-1 docs |
| **1 ✅ DONE** | `lshaped.py` — `solve_lshaped()`: builds the probability-weighted extensive form, annotates the first stage, and drives `solve_benders`. **Probability weighting rides in the objective** (Expectation scales each recourse cost by `p_s`), so the engine is unchanged — the "missing primitive" needed no engine edit. | the L-shaped **decomposition structure** is verified via the Python-side `detect_decomposition` — first-stage complicating vars, one separable recourse block per scenario; the objective equals the probability-weighted extensive form; risk-neutral gate (CVaR L-shaped refused) — **4 tests pass, ruff clean**. Certified `L-shaped == extensive-form` solve → CI (needs Rust+pounce). | notebook: farmer solved both ways (Phase-1 docs); `example_farmer_*` gallery |
| **2 ✅ DONE** | `ph.py` — `progressive_hedging()` with an **injected subproblem solver** (PH loop is pure NumPy given the solve); convex-nonlinear recourse via `solve_lshaped(method="gbd")` → `solve_gbd` | PH converges to the expected-value first-stage `Σ p_s a_s` on analytic separable-quadratic subproblems across ρ∈{0.1,1,5}, keeps the `Σ p_s w_s = 0` invariant, and drives **both** the primal (dispersion) and dual (consensus-movement) residuals to 0 — the dual residual was a real convergence fix the ρ-sweep caught; method routing validated — **8 tests pass, ruff clean**. Real-NLP subproblem solve + GBD recourse solve → CI. **Advisor `SCENARIO`/PH generator deferred**: the advisor's `StructureReport` has no scenario/probability awareness, so a generator has nothing to fire on until the advisor learns scenario structure (a separate, larger change); the `MethodKind.PROGRESSIVE_HEDGING` slot stays reserved and `solve_lshaped`/`progressive_hedging` are the direct drivers. | notebook: PH + CVaR/chance examples (docs iteration) |
| **3** | multistage (nested Benders / SDDP slot), integer L-shaped (Laporte–Louveaux), SAA confidence intervals, DRO bridge to `ro/` | multistage farmer/energy instances; integer-recourse instances certified; SAA lower/upper statistical bounds reported | multistage notebook section; `docs/references.bib` entries; **`jupyter-book build docs/` zero warnings** |

Every phase ships fails-before/passes-after tests per the merged loop protocol
(fast, class-not-instance). Each decomposition method is validated **bound-neutrally**
against the extensive form (same certified objective), and the probability-weighting
engine change carries a dedicated differential test.

## 6. Tests & acceptance (concrete)

- **Extensive form vs analytic:** newsvendor critical-fractile closed form; the
  Birge–Louveaux farmer problem (known first-stage plan + expected profit).
- **L-shaped ≡ extensive form:** identical certified objective and first-stage `x`
  on shared scenarios; verified across scenario counts.
- **Probability weighting:** the summed cut/bound reproduces `Σ_s p_s Q_s(x)` at a
  fixed `x` (direct, sub-second).
- **CVaR / chance constraint:** vs brute-force empirical CVaR / coverage on small
  scenario sets; the `ε`-coverage of the returned `x` matches the SAA target.
- **Refusals:** integer recourse on `lshaped` → routed to extensive-form MINLP or
  refused with a clear message; multistage in v1 → refuse; probabilities not summing
  to 1 → refuse.

## 7. Documentation & examples

Per the repo's Jupyter Book policy (CLAUDE.md → *Documentation*), the module ships
user-facing docs, worked examples, and citations that build with **zero warnings**,
and the examples are pinned by a test so they cannot rot.

- **Notebook** `docs/notebooks/stochastic_programming.ipynb` (single source of truth):
  two-stage SP with recourse, scenarios vs SAA sampling, the **newsvendor** (closed
  form, to build intuition) and the **Birge–Louveaux farmer problem** solved end-to-end
  — first as the extensive form, then via the L-shaped method, showing the *identical*
  certified optimum (the "extensive form is the oracle" story made visual) and a
  scaling comparison. Further cells: a risk-averse **CVaR** variant and a
  **chance-constrained** variant, each contrasted with the risk-neutral plan. Every
  markdown cell carries `{cite:p}`/`{cite:t}` MyST citations.
- **Bibliography** — add to `docs/references.bib`: Birge & Louveaux (2011),
  Kall & Wallace (1994), Van Slyke & Wets (1969, L-shaped), Rockafellar & Uryasev
  (2000, CVaR), Rockafellar & Wets (1991, progressive hedging), and Shapiro–Dentcheva–
  Ruszczyński (2021).
- **TOC** — register the notebook in `docs/_toc.yml` under the appropriate part.
- **Gallery examples** — runnable `example_farmer_*` / `example_newsvendor_*` models
  added to the pure-modeling gallery, covered by the whole-gallery `validate()` smoke
  test (the E1–E3 pattern from PR #439) so the documented examples are built and
  validated on every CI run.
- **API docstrings** — `TwoStageProblem`, `ScenarioSet`, `expectation`, `cvar`,
  `chance_constraint` carry runnable, doctest-style examples (the §3.4 snippet is the
  seed).
- **Build gate** — `jupyter-book build docs/` completes with zero warnings.

Documentation is incremental across the phases (the "Docs" column in §5): Phase 0
ships the notebook skeleton + newsvendor; Phase 1 adds the farmer problem solved both
ways; Phase 3 completes it and enforces the zero-warning build.

## 8. SOTA positioning

Comparable tools: **mpi-sppy / PySP** (Pyomo two-/multistage, PH + L-shaped),
**SDDP.jl** (multistage SDDP), **StochasticPrograms.jl**. discopt's differentiators:
(a) the SP is solved by the *same* decomposition engine as deterministic problems,
with a **global MINLP + certificate** (`gap_certified`), where most SP tools assume
LP/QP subproblems; (b) **convex-nonlinear recourse** via GBD, not just LP recourse;
(c) it **composes** with the rest of discopt — CVaR/chance via the GDP layer, and a
natural DRO bridge to the shipped `ro/` module. Gaps vs the leaders: no SDDP/
multistage or scenario reduction/importance sampling in v1 (Phase 3), and PH is a
primal heuristic (dual bound via Lagrangian), not a global method on its own.
