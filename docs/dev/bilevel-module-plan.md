# Bilevel Optimization Module — Design & Implementation Plan

**Date:** 2026-07-03
**Status:** design plan for a new `python/discopt/bilevel/` module.
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
  problem.py             # BilevelProblem: the user-facing front-end + .formulate()
  symbolic_diff.py       # ∂Expression/∂Variable -> Expression  (the new engine piece)
  kkt.py                 # KKT reformulation: stationarity + feas + complementarity
  strong_duality.py      # LP strong-duality reformulation
  convexity_gate.py      # lower-level-convex-in-y check (wraps the certifier)
```

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
| **1** | `strong_duality.py` + `BilevelProblem` for **LP–LP** bilevel; convexity/integer gates | Bard's linear bilevel examples and the classic linear bilevel (known optima) certified correct; KKT and strong-duality agree on LP lower levels | `docs/notebooks/bilevel.ipynb` skeleton + first worked LP-bilevel example with `{cite:p}` citations |
| **2** | `kkt.py` for **convex-QP / convex-NLP** lower level | Toy convex-lower-level instances certified against a brute-force/analytic follower response; `mpec` `gdp` path gives `gap_certified=True` | notebook: convex-lower-level example; `example_bilevel_*` added to the modeling gallery |
| **3** | Advisor hook; MI-upper-level examples (facility interdiction, toll-setting) | end-to-end gallery examples with `validate()` + pinned optima; refusal tests for nonconvex/integer/pessimistic | notebook complete; `_toc.yml` entry; `docs/references.bib` entries; **`jupyter-book build docs/` zero warnings** |

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
