# Bilevel Optimization Module — Design & Implementation Plan

**Date:** 2026-07-03 (audited & corrected 2026-07-13)
**Status:** implemented (`python/discopt/bilevel/`), Phases 0–3 landed. A 2026-07-13
audit found and fixed a P0 false-optimal in the certified KKT path — see the
"Audit correction" block in §6. The sound end-to-end path is `strong_duality`.
**Thesis:** a bilevel program is solved by replacing the *follower's* optimization
with its optimality conditions, collapsing the two levels into one. discopt already
has the two hardest ingredients — the **complementarity/MPEC machinery**
(`discopt.mpec`: Scholtes / SOS1 / GDP) and **autodiff over the expression DAG** — so
bilevel is roughly **a front-end + one new engine component** (a *symbolic*
differentiator), not a new solver. This is the natural companion to the robust and
MPEC layers already shipped.

---

## 0. What bilevel is, and why discopt is close

A bilevel (Stackelberg) program is

```
  min_{x,y}   F(x, y)                     (leader / upper level)
  s.t.        G(x, y) <= 0
              y ∈ argmin_{y'} { f(x, y') : g(x, y') <= 0 }   (follower / lower level)
```

The leader chooses `x`; the follower responds by optimizing `f` over `y`. The
argmin constraint is the whole difficulty — it is an *optimization inside a
constraint*.

**The standard exact reduction (for a lower level that is convex in `y` for fixed
`x`):** replace `y ∈ argmin{…}` by the follower's **KKT conditions**, which are then
*necessary and sufficient* for lower-level optimality. That yields a single-level
**MPEC** — exactly the problem class `discopt.mpec` already solves. So:

> **Bilevel = KKT-reformulation front-end → `discopt.mpec`.**

The mapping (verified against the code) is:

| KKT ingredient | discopt asset | Status |
|---|---|---|
| Stationarity `∇_y f + Σ μ_i ∇_y g_i = 0` | expression DAG + a **symbolic ∂/∂y** | **must build** (numeric autodiff exists, symbolic does not) |
| Primal feasibility `g(x,y) <= 0` | `m.subject_to(g <= 0)` | exists |
| Dual feasibility `μ >= 0` | `m.continuous(name, lb=0)` | exists |
| Complementarity `0 <= μ ⊥ −g >= 0` | `discopt.mpec.complementarity` / `solve_mpec` | exists |
| Convexity check on the lower level | `discopt/_jax/convexity` certifier | exists (reuse) |
| In-place `.formulate()` builder pattern | `ro.counterpart.RobustCounterpart` | exists (mirror) |

The single missing engine piece is a **symbolic differentiator** that maps an
`Expression` to the `Expression` for its partial derivative w.r.t. a `Variable`.
discopt today differentiates only *numerically* (JAX `jax.grad`/`jax.jacfwd` over
`compile_expression_params`). A numeric gradient can be embedded as a `CustomCall`
node, but `CustomCall` is **AD-only** — it loses the global certificate and rejects
integer variables — so it is unusable for a certifiable global bilevel solve. A
symbolic ∂/∂y that returns ordinary DAG `Expression`s keeps the whole reformulated
model inside the certified global MINLP path. Building it is the crux of Phase 0,
and it is **reusable well beyond bilevel** (exact sensitivity, symbolic Jacobians
for cuts, teaching).

---

## 1. Scope (v1) and explicit non-goals

**In scope (v1):**
- **Optimistic** bilevel (the leader breaks follower ties in its own favor — the
  standard, well-posed, tractable case).
- **Lower level convex in `y`** for each fixed `x` (linear, convex-quadratic,
  convex-nonlinear). KKT is then exact.
- Two reformulations, user-selectable:
  1. **KKT–MPEC** (general convex lower level) → `discopt.mpec`.
  2. **Strong-duality** (LP lower level only): replace the lower problem by
     `primal feasible ∧ dual feasible ∧ cᵀy = bᵀλ`. No complementarity product;
     sometimes tighter and integer-upper-friendly. (Classic for linear bilevel.)
- **Mixed-integer *upper* level** is fine (leader integrality does not break the
  reduction). Solved globally via the MPEC's SOS1/GDP path.

**Out of scope in v1 — and refused loudly, not silently approximated** (CLAUDE.md §3):
- **Nonconvex lower level** — KKT is only *necessary*, so the reduction is unsound
  (it can certify a non-follower-optimal `y`). Gate with the convexity certifier;
  raise `NotImplementedError` naming the offending lower-level terms.
- **Integer *lower-level* variables** — KKT/strong-duality do not characterize
  integer optima; the follower's value function is discontinuous. Refuse (a future
  value-function / branch-and-bound-in-the-lower-level approach is a separate design).
- **Pessimistic** bilevel (leader hedges against adversarial follower ties) — a
  genuinely harder problem (min–max structure); Phase 3+ research.

---

## 2. Module layout

```
python/discopt/bilevel/
  __init__.py            # BilevelProblem, exports
  problem.py             # BilevelProblem: front-end + .formulate()/.build_kkt_system()
                         #   AND the convexity gate (folded in; no separate file)
  symbolic_diff.py       # ∂Expression/∂Variable -> Expression  (the new engine piece)
  kkt.py                 # KKT reformulation: stationarity + feas + complementarity
  strong_duality.py      # LP/QP strong-duality reformulation
```

> **As-built note (audit 2026-07-13):** there is no separate `convexity_gate.py`;
> the lower-level-convex-in-y check lives in `problem.py` (`_convexity_status` /
> `_hessian_in_y` / `_gate_convexity`), a constant-Hessian PSD test covering LP and
> convex-QP. The general convexity certifier is not yet wired in (non-quadratic
> convex lower levels are refused loudly). `problem.py` also exposes
> `build_kkt_system()` — the sound KKT-math construction, separate from the
> encoding choice in `formulate()`.

Optional dep: none beyond the core (mpec + _jax are already core). Mirrors the flat,
builder-pattern shape of `ro/`.

---

## 3. The symbolic differentiator (`symbolic_diff.py`) — the crux

**Contract:** `diff(expr: Expression, wrt: Variable, *, model) -> Expression`, and a
vectorized `grad(expr, wrt_vars) -> list[Expression]`. Pure DAG-to-DAG; the result
routes through the normal global/certified path (no `CustomCall`).

**Rules over the known node types** (`core.py`): `Constant→0`; `Variable→1` if it is
`wrt` (elementwise for array vars / index match) else `0`; `BinaryOp` for `+,-` linear,
`*` product rule, `/` quotient rule, `**` power rule (constant exponent; general
`u**v` via `u**v·(v'·ln u + v·u'/u)` guarded for `u>0`); `UnaryOp` neg/abs (abs'
via `sign`, flagged nonsmooth); `FunctionCall` chain rule with a derivative table
(`exp'→exp`, `log'→1/·`, `sqrt'→1/(2√·)`, `sin'→cos`, …); `SumExpression`/
`SumOverExpression` linearity; `IndexExpression` component-wise; `MatMulExpression`
matrix product rule. Simplify trivially (`0·e=0`, `1·e=e`, `e+0=e`) to keep the
emitted stationarity system small.

**Validation (the acceptance gate):** a **differential test** against JAX — for
random expressions over random points, `symbolic_diff` compiled through
`compile_expression_params` must equal `jax.grad(compile_expression_params(expr))` to
machine tolerance. This is the same "measurement wins" discipline used in the review
fixes; the symbolic gradient is only trusted once it matches the numeric one across a
fuzz. Nonsmooth points (`abs`, `**` at 0) are excluded from the tolerance check and
documented.

---

## 4. User-facing API (`problem.py`)

Mirrors `RobustCounterpart`'s in-place `.formulate()` builder (verified pattern):

```python
from discopt.bilevel import BilevelProblem

bl = BilevelProblem(
    model,                       # the discopt Model (holds upper vars/constraints/obj)
    upper_vars=[x],              # leader decisions
    lower_vars=[y],              # follower decisions
    lower_objective=f_expr,      # follower minimizes this (Expression)
    lower_constraints=[...],     # follower's g(x,y) <= 0  (list[Constraint])
    lower_sense="min",           # follower sense
)
bl.formulate(method="kkt")       # or method="strong_duality" (LP lower level)
result = model.solve(...)        # ordinary global MINLP solve of the resulting MPEC
```

`formulate()` (guarded against double-call, exactly like `RobustCounterpart`):
1. **Convexity gate** — certify `lower_objective` and each `lower_constraint` convex
   in `lower_vars` (holding `upper_vars` fixed, treating them as parameters). Refuse
   loudly otherwise. Refuse if any `lower_var` is integer.
2. **Dispatch** to `kkt` or `strong_duality`.
3. Leave the model as a single-level MPEC that `model.solve()` handles.

**KKT path (`kkt.py`):** for lower level `min_y f s.t. g_i(x,y) <= 0`:
- mint multipliers `μ = m.continuous(f"{prefix}_mu{i}", lb=0)` per lower constraint
  (per row for vector bodies);
- build the Lagrangian `L = f + Σ μ_i g_i` and emit **stationarity**
  `subject_to(symbolic_diff(L, y_k) == 0)` for each lower variable `y_k` (this is
  where `symbolic_diff` is essential);
- primal feasibility `g_i <= 0` is already on the model (or re-added);
- **complementarity** `mpec.complementarity(μ_i, -g_i)` for each pair, handed to the
  chosen MPEC method (`gdp`/`sos1` for a global certificate, `scholtes` for a fast
  local NLP).
- Equality lower-level constraints `h(x,y)=0` get free-sign multipliers (two
  nonnegative parts or an unbounded var) and appear only in stationarity, no
  complementarity.

**Strong-duality path (`strong_duality.py`, LP lower level):** for
`min_y cᵀy s.t. A y <= b − Bx, y >= 0`, emit lower primal feasibility, the LP dual
`Aᵀλ >= c, λ >= 0`, and the strong-duality equality `cᵀy == (b − Bx)ᵀλ` (bilinear in
`x`·`λ`, handled by the global McCormick/spatial path). No complementarity product.
Coefficient extraction reuses the same expression-walk idioms `ro/polyhedral.py`
already uses.

---

## 5. Soundness (the non-negotiables)

1. **KKT/strong-duality are sound only for a convex lower level.** The convexity
   gate is a hard precondition, not advice — a nonconvex lower level makes the
   single-level model a *different* (wrong) problem. Reuse the certifier that PR #421
   confirmed sound; refuse when it cannot certify convexity.
2. **Optimistic semantics are documented and asserted.** The KKT reduction yields the
   optimistic solution; the doc and API say so. A pessimistic request raises.
3. **Integer lower variables are refused** (KKT invalid).
4. **The complementarity solve inherits `mpec`'s soundness contract**: `sos1`/`gdp`
   give a global certificate; `scholtes` is a documented *local* NLP homotopy
   (labeled, `gap_certified=False`). Default to `gdp` for certifiable answers.
5. **`symbolic_diff` is trusted only after the differential test passes** — an
   incorrect stationarity equation silently changes the follower's optimum, so the
   gradient engine is fuzzed against JAX before any bilevel test is believed.

---

## 6. Phased plan

| Phase | Deliverable | Acceptance | Docs |
|---|---|---|---|
| **0 ✅ DONE** | `symbolic_diff.py` + differential-test harness (`test_bilevel_symbolic_diff.py`) | symbolic ∂/∂x matches `jax.grad` to 1e-9 over a fuzz of random DAGs (all node types); nonsmooth points documented/excluded — **29 tests pass, ruff clean** | module + function docstrings with runnable examples; the differential test as executable documentation |
| **1 ✅ DONE** | `BilevelProblem` (`problem.py`) + `kkt.py` — the **KKT reformulation** for an LP-in-`y` lower level; integer/nonconvex/pessimistic gates | the emitted stationarity + primal + dual + complementarity conditions exactly characterize follower optimality — validated against an independent **scipy follower-LP oracle** (min & max followers, several leader `x`), stationarity shown to bind, all gates enforced — **11 tests pass, ruff clean**. End-to-end *certified* solve of Bard's instances → CI (needs Rust+pounce). | module/API docstrings with a worked toll example (the seed for the notebook) |
| **2 ✅ DONE** | `strong_duality.py` (aggregate-complementarity / strong-duality reformulation) + convexity gate **lifted to convex-QP** (PSD constant Hessian in y) | KKT ≡ strong-duality verified equivalent at the follower optimum (scipy LP oracle); convex-QP lower level accepted with KKT correct (QP oracle); nonconvex / non-quadratic / nonlinear-equality lower levels refused — **8 tests pass, ruff clean**. Convex-**NLP** (non-quadratic) via the full convexity certifier is deferred (gate refuses loudly, names it). End-to-end certified solve → CI. | module/API docstrings; worked LP + convex-QP examples seed the notebook |
| **3 ✅ DONE** | `example_bilevel_toll` gallery model (convex-QP toll setting) + **follower-variable-bound correctness fix** (finite bounds folded into the KKT); `docs/notebooks/bilevel.ipynb`; `references.bib` + `_toc.yml` | the gallery example builds + `validate()`s in the whole-gallery smoke test; notebook code cells execute solver-free (formulate is pure-Python); bound-active regression test (a follower optimum on a bound is excluded without the fix) — **bilevel suite + gallery green (ruff clean)**. `jupyter-book build` zero-warning check → CI (not installed locally). Advisor hook deferred (bilevel does not map onto the decomposition advisor). | notebook with `{cite:p}` citations (Bard/Dempe/Colson–Marcotte–Savard/Kleinert et al.); gallery example; TOC + bib entries |

**Audit correction (2026-07-13) — the certified KKT solve never actually worked.**
The Phase-1/3 "End-to-end *certified* solve → CI" line was aspirational: **no test
ever called `model.solve()` on a `BilevelProblem`** (the phase suites validate the
`formulate()` output against a scipy oracle; the gallery example is only
`validate()`d). An end-to-end solve exposed a **P0 false optimal**: the module
default `method="kkt", mpec_method="gdp"` on a linear-follower bilevel certified a
follower-*infeasible* point (`gap_certified=True`) because the KKT multipliers are
unbounded and the GDP big-M treated discopt's ±1e20 unbounded sentinel as a valid
(but numerically vacuous) `M`, so the complementarity disjunction was never
enforced. Fixes:
- **Shared GDP layer** (`_jax/gdp_reformulate.py`): `_compute_big_m` now refuses a
  sentinel-magnitude bound (`|bound| ≥ _BIGM_SENTINEL = 1e15`) as it already did for
  ±inf and as SOS1 / `_compute_big_m_lp` already did — a sentinel `M` is not a
  usable big-M. (Real finite bounds below the sentinel still use the true bound,
  preserving the #413 "don't shrink a valid `M`" fix.)
- **Bilevel front-end**: `formulate(method="kkt", mpec_method="gdp"|"sos1")` now
  **refuses loudly** when the follower multipliers are unbounded (the common case),
  pointing at `method="strong_duality"`.
- **The working certified-ish path is `strong_duality`** (a single bilinear
  equality, no big-M): it solves the linear/convex-QP bilevel to the true
  optimistic optimum (the new `test_bilevel_phase3.py` pins this end-to-end and
  checks follower optimality against a scipy oracle). Its solve is *not*
  gap-certified because the strong-duality equality is nonconvex — an honest state,
  not the "certified" the table claimed. A genuinely gap-certified KKT path needs
  valid finite multiplier bounds (future work), not a big-M over the sentinel.

**Re-scope note (2026-07-03, after Phase 0 landed).** The original plan put LP
strong-duality in Phase 1 and the KKT reformulation in Phase 2. With Phase 0's
symbolic differentiator done and tested, the **KKT path became the lower-risk, higher-
reuse next step**: it consumes `symbolic_diff` directly, needs no separate linear-
coefficient extraction, handles LP *and* (later) convex-nonlinear lower levels
uniformly, and — critically — its correctness is checkable **without the global
solver** via a scipy follower-LP oracle (the strong-duality equality is bilinear and
only meaningfully testable end-to-end). So Phase 1 now delivers `BilevelProblem` + the
KKT reformulation for LP lower levels; **strong-duality moves to Phase 2** as the
LP-case cross-check, alongside lifting the affine-in-`y` gate to the full convexity
certifier for convex QP/NLP. Reality-drove-the-plan, recorded per CLAUDE.md §4.

Each phase ships with fails-before/passes-after tests per the merged loop protocol
(fast, class-not-instance), and each reformulation is a **bound-changing** change, so
it carries the differential regime: the reformulated single-level optimum must equal
an independent oracle (brute-force follower response on small instances) and never
cut a true bilevel-feasible leader point.

## 7. Tests & acceptance (concrete)

- **`symbolic_diff`**: 3–4k random-DAG × random-point trials vs `jax.grad`; product/
  quotient/chain/power rules each pinned; array/index/`dm.sum` covered.
- **LP bilevel**: Bard (1998) Example 5.1.1 and the standard linear bilevel with a
  known optimum; assert leader objective and that `y` is genuinely the follower's
  optimum (recompute the lower LP at the leader `x`).
- **Convex-QP bilevel**: toy market/pricing instance where the follower response has
  a closed form; assert the certified `(x,y)` matches.
- **Refusals**: nonconvex lower objective → `NotImplementedError`; integer lower var
  → refuse; `method="pessimistic"` → refuse. Each a fast test.
- **KKT ≡ strong-duality** on LP lower levels (same optimum, both methods).

## 8. Documentation & examples

Per the repo's Jupyter Book policy (CLAUDE.md → *Documentation*), the module is not
"done" until it ships user-facing docs, worked examples, and citations that build
with **zero warnings** — and the examples are pinned by a test so they cannot rot.

- **Notebook** `docs/notebooks/bilevel.ipynb` (the single source of truth for the
  content): motivation (Stackelberg leader–follower games), the KKT reduction and its
  convexity precondition, the optimistic-vs-pessimistic distinction, then a worked
  **toll-setting / network-interdiction** model solved end-to-end with
  `BilevelProblem(...).formulate()` → `model.solve()`, showing the recovered leader
  decision *and* a check that the follower response is genuinely lower-level-optimal.
  A second cell demonstrates the `symbolic_diff` engine directly (`diff`/`grad`) and
  the differential-test idea. Every markdown cell carries `{cite:p}`/`{cite:t}` MyST
  citations.
- **Bibliography** — add entries to `docs/references.bib`: Bard (1998), Dempe (2002),
  Colson–Marcotte–Savard (2007), and the Kleinert–Labbé–Ljubić–Schmidt (2021) survey.
- **TOC** — register the notebook in `docs/_toc.yml` under the advanced-topics part.
- **Gallery example** — a runnable `example_bilevel_*` model added to the pure-modeling
  gallery, covered by the whole-gallery `validate()` smoke test (the E1–E3 pattern from
  PR #439) so the documented example is built and validated on every CI run.
- **API docstrings** — `BilevelProblem`, `diff`, `grad`, and each reformulation carry
  runnable, doctest-style usage examples (the API snippets in §3–§4 are the seed). The
  Phase 0 `symbolic_diff` module already ships this.
- **Build gate** — `jupyter-book build docs/` completes with zero warnings.

Documentation is incremental across the phases, not a Phase-3 afterthought (the "Docs"
column in §6): Phase 0 already ships full `symbolic_diff` docstrings + the differential
test as executable documentation; Phase 1 adds the notebook skeleton with the first
worked LP-bilevel example; Phase 3 completes the notebook, the gallery example, and the
zero-warning build.

## 9. SOTA positioning

Comparable tools: **Pyomo.PAO** (linear/quadratic bilevel, KKT + strong-duality),
**YALMIP** (KKT via `solvebilevel`), **MibS** (mixed-integer bilevel, branch-and-cut),
Gurobi's bilevel examples. discopt's differentiators: (a) the KKT-MPEC is solved by a
**global** MINLP with a **certificate** (SOS1/GDP → `gap_certified`), where most
tools rely on big-M with unproven bounds; (b) a **symbolic** stationarity builder
that handles convex-nonlinear lower levels, not just LP/QP; (c) it composes with the
rest of discopt (robust, GDP, NN embedding). Gaps vs MibS: no mixed-integer *lower*
level (the genuinely hard frontier) — explicitly deferred, not faked.
