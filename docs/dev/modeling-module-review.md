# Modeling Module Review — Correctness, Thoroughness, Performance, and SOTA Assessment

**Date:** 2026-07-03
**Scope:** `python/discopt/modeling/` (`core.py`, `sets.py`, `indexed.py`, `latex.py`,
`implicit.py`, `gams_parser.py`, `examples.py`, `__init__.py`) and its direct
consumers where the modeling layer's contracts are enforced (notably
`python/discopt/_jax/problem_classifier.py`).
**Method:** Full read of `core.py`/`sets.py`/`indexed.py`/`latex.py`/`implicit.py`
(~4,800 lines) with every suspected defect reproduced end-to-end against the installed
package; `gams_parser.py`, `examples.py`, and a second pass over `latex.py`/`implicit.py`
reviewed by delegated verification agents whose top findings were reproduced with
runnable `.gms`/Python inputs (spot-checked here).
Baseline before review: the modeling-adjacent fast suite is green
(152 passed, 1 skipped: `test_sets*`, `test_indexed_*`, `test_model_latex`,
`test_implicit_node`).

Findings marked **[CONFIRMED]** have a runnable failing repro; **[BY INSPECTION]** are
unambiguous from control flow.

---

## 1. Summary of findings

| # | Severity | Component | Finding |
|---|----------|-----------|---------|
| M1 | **P0 correctness** | `_jax/problem_classifier.py` (contract defined by `modeling/core.py`) | Vector (broadcast) constraints are silently **aggregated into one summed row** on the LP/MILP algebraic extraction path → solver returns an **infeasible point certified "optimal"** [CONFIRMED] |
| M2 | **P0 correctness** | `core.py:3630` (`from_nl`) | Binary variables are re-created with default `[0,1]` bounds, **discarding the parsed `.nl` bounds** — a fixed binary silently un-fixes; affects `from_nl` and the `from_pyomo` round-trip [CONFIRMED] **✅ RESOLVED (X-3, #413): `from_nl` now stamps the parsed lb/ub onto the binary, clamped into `[0,1]`; regression `TestFromNlBinaryBoundsX3` in `test_nl_reconstruction.py`.** |
| M3 | **P0 correctness** | `core.py` (no ownership check) | Expressions mixing variables from **two different models** are silently accepted; the foreign variable **aliases by flat index** onto a same-index variable of the solved model → silently wrong answer [CONFIRMED] |
| M4 | P1 | `core.py:157` | `Expression.__eq__` returns a `Constraint`, so `expr in seq` is **always True**, `Constraint == Constraint` is always True, and all non-`Variable` expressions are unhashable [CONFIRMED] |
| M5 | P1 | `core.py:1910,1933` | `subject_to` **mutates** the passed `Constraint` in place (`c.name = name`); re-adding the same object renames the earlier row; duplicate constraint names are never validated though `constraint_duals` is keyed by name [CONFIRMED] |
| M6 | P1 | `core.py:solve` | Misspelled `solve()` keyword arguments are **silently swallowed** (`gap_tolerence=…` accepted without error) [CONFIRMED] |
| M7 | P2 perf | `core.py:3213` | `_check_name` rebuilds the full name set per declaration → **O(n²)** model build; measured 37→94→170 µs/var at n=1k/2k/4k [CONFIRMED] |
| M8–M12 | P2 | various | Robustness/API gaps (§4) |
| G1–G15 | **P0–P2** | `gams_parser.py` | Fifteen verified defect classes; the five worst silently build a *different model* than the GAMS source and let the solver certify it (§6.1) |
| E1–E3 | **P0/P1** — ✅ RESOLVED | `examples.py` | Haverly pooling example is mathematically wrong (certified optimum 1390 vs the classic 400); `example_logical_constraints` crashes (never built by any test); `example_reactor_design` is provably infeasible (§6.2) |
| L1–L8 | P1–P3 | `latex.py` | `to_latex`/Jupyter `_repr_` **crashes** on any model with `if_then`/`either_or`/`logical`; fast-path constraint families render as zero constraints; several precedence/escaping defects (§6.3) |
| I1–I2 | P2 | `implicit.py` | Absolute-only Newton tolerance falsely NaN-fails well-posed scaled residuals; build-time probe uses wrong `u` length for vector inputs (§6.4) |

Checked and found **sound** (no action): the constraint-normalization convention
(`__ge__` → `rhs − lhs <= 0`) and its LaTeX un-normalization; `latex.py` binary-operator
precedence for the cases `-x**2`, `(x**2)**3`, `a-(b-c)` (the *non*-BinaryOp nodes do
have precedence bugs — see L4 in §6.3);
`sets.py` set algebra, dedup, dimen validation, `ProductSet.ordinal`;
`indexed.py`'s `affine_form` (conservative: anything non-single-variable-affine
returns `None` and falls back — parameters correctly stay symbolic);
`implicit.py`'s Newton/IFT node (nonsingularity and convergence gates propagate NaN
rather than returning a wrong root; `custom_root` keeps higher-order AD);
`if_else`'s hull-forced disjunction with interval-derived sound auxiliary bounds;
`SolveResult.__post_init__`'s gap-certification downgrade guard; the documented
huge-bounds `UserWarning` (verified it actually fires).

---

## 2. P0 correctness bugs

### M1. Vector constraints silently collapse to one summed row on the LP/MILP path

**The single most serious finding of this review.** The modeling layer's contract —
implemented correctly by `NLPEvaluator` and by the autodiff LP extractor — is that a
`Constraint` whose body is array-shaped means **one row per component**. The *algebraic*
fast-path extractor (`_extract_linear_coefficients`,
`_jax/problem_classifier.py:219-226`) instead assumes every body is scalar and contains
this branch for a bare array variable:

```python
else:
    # Array variable treated as sum when used as scalar
    for j in range(node.size):
        c[offset + j] += scale
```

So `a + b >= 1` with `a, b` of shape (2,) is extracted as the single row
`sum(a) + sum(b) >= 1`, and because the extraction *succeeds*, the correct autodiff
fallback (`_extract_lp_data_autodiff`, whose docstring explicitly handles vector
bodies row-per-component) never runs.

**Reproductions (all return status `"optimal"`):**

- LP: `min Σa+Σb s.t. a+b >= 1` (shape (2,), boxes [0,1]) → returns
  `a=b=[0.25,0.25]`, objective **1.0**; true optimum **2.0**. The model's own
  `NLPEvaluator.evaluate_constraints` at the returned point gives `[0.5, 0.5] > 0` —
  the certified point is **infeasible**.
- Vector **equality** `a + b == 1`: same wrong answer (1.0).
- **MILP**: set-cover `min Σy+Σz s.t. y+z >= 1` (shape (3,) binaries) → objective
  **1.0** with `y=[1,0,0], z=[0,0,0]`; true optimum **3.0**.
- Control: the identical model written as a scalar loop
  (`[y[i]+z[i] >= 1 for i in range(3)]`) solves correctly on the same path.
- QP end-to-end probes came out correct (an asymmetric instance verified feasible and
  optimal), but the quadratic walker has the **same array-as-sum branch**
  (`problem_classifier.py:372-376`) and must be fixed/audited identically.
- The DAE module's vector equalities escape only by accident: their `MatMul` bodies
  raise `_NotLinearError`, routing those models to the row-correct path.

This violates the repo's core invariant — `incorrect_count ≤ 0` — for any LP/MILP a
user writes in the natural numpy-vectorized style the API advertises. A related
consequence of the same scalar-body assumption: a **shape-mismatched** body
(`a(3,) + b(2,)`) does not raise at build time and also solves "optimal" on this path
(the NLPEvaluator at least raises a JAX broadcast error; the algebraic path doesn't
even do that).

**Fix (two stages):**
1. *Minimal, provably safe:* in both `_extract_linear_coefficients` and
   `_extract_quadratic_coefficients`, **raise** `_NotLinearError` /
   `_NotQuadraticError` whenever a node's value is array-shaped in a scalar position
   (bare array `Variable`, array `Constant`, broadcasted results). Affected models
   then route to the autodiff extractor, which is documented-correct. This is the
   "hard right fix over the band-aid" — deleting the wrong convenience branch, not
   patching around it.
2. *Performance follow-up:* teach the algebraic extractor to expand array bodies into
   proper rows (shape inference through `+`, `-`, scalar `*`, broadcast), restoring
   the fast path for vectorized models. Verify bound-neutrally: extracted `(A, b, c)`
   must match the autodiff extractor exactly on a panel of vectorized models.

**Regression tests:** the three repros above (LP, vector-equality, MILP), each
asserting both the objective value and elementwise feasibility of the returned point;
plus a property test comparing algebraic vs autodiff extraction row counts and
matrices on randomly generated vectorized affine models.

### M2. `from_nl` discards binary variable bounds — fixed binaries un-fix

`core.py:3630`:

```python
elif vt == "binary":
    m.binary(name, shape=shape)     # lb_vals/ub_vals read but ignored
```

Continuous and integer variables reconstruct their parsed bounds; binaries are reset
to `[0,1]`. A `.nl` file with a fixed binary (`lb = ub = 1` — routine output of
presolve, fixing heuristics, or user `.fx`) silently re-opens the variable.

**Reproduction:** model with binary fixed to 1 (optimum 11.0) → `to_nl` → the file and
the Rust parser both correctly carry `lb=ub=1` → `from_nl` → re-imported model solves
to **2.0** with the binary at 0. Silently wrong on round-trip; also poisons
`from_pyomo`, which is implemented as a `.nl` round-trip.

**Fix:** pass the parsed bounds through: `m.binary(...)` then clamp
`var.lb/var.ub` to the parsed values (or extend `Model.binary` to accept bounds).
Intersect with `[0,1]` for soundness. Regression test: `.nl` round-trip of a model
with a fixed binary must preserve the optimum; also parse a hand-written `.nl` with
`b`-section bounds `1 1` on a binary column.

**✅ RESOLVED (X-3, #413).** Implemented exactly as prescribed: after
`var = m.binary(name, shape=shape)`, `from_nl` sets
`var.lb = np.broadcast_to(np.clip(lb, 0, 1), shape)` and likewise for `ub`
(the clip keeps a binary column inside `[0,1]`; the broadcast handles array
binaries). Reproduced on `origin/main` (a `min x + 10·y` fixture with `y` fixed to 1
round-tripped to `~0.0` with `y=0` instead of `10.0`); after the fix the reconstructed
`y` carries `lb=ub=1` and the solve returns `10.0`. Regression:
`TestFromNlBinaryBoundsX3` in `python/tests/test_nl_reconstruction.py` — a `to_nl`→
`from_nl` round-trip (asserts bounds survive + correct solve, `@pytest.mark.smoke`) and
a writer-emitted `.nl` re-parse (asserts the `b`-section `1 1` binary bound survives).
Fails-before verified by stashing the core fix. Closes infra INT-2 (same root).

### M3. Cross-model expressions are silently accepted and alias by index

No layer checks that variables in an objective/constraint belong to the model being
solved. Variables carry a flat `_index` local to their own model, so a foreign
variable **aliases whatever variable of the solved model has the same index**.

**Reproduction:** `ma.minimize(xa + xb)` with `xb` from model B: solves "optimal",
objective 1.0 — `xb` (index 0 in B) silently aliased to `xa` (index 0 in A); the
result contains no `xb` at all. In real use (copy-paste between model-building
functions, notebook re-runs re-creating models) this yields a *plausible-looking wrong
answer*, not an error.

**Fix:** `Model.validate()` (already called by `solve()`) walks objective and
constraint DAGs once, asserting every `Variable`/`Parameter` has `node.model is self`
— raise with the variable name and the owning model's name. JuMP and gurobipy both
raise immediately on this; it is table stakes. Cost is one O(DAG) walk per solve.
Regression test: the repro above must raise `ValueError`.

---

## 3. P1 correctness/robustness

### M4. `Expression.__eq__` breaks membership, equality, and hashing

`expr == other` returns a `Constraint` (required for the modeling DSL — same choice
as Pyomo/cvxpy), but the consequences are unmitigated:

- `u in [v]` → **True** for any expressions u, v (Python evaluates `__eq__` and
  truth-tests the returned `Constraint`, which is truthy). [CONFIRMED]
- `Constraint == Constraint` → always True (dataclass-generated `__eq__` tuple
  comparison routes through `Expression.__eq__`). [CONFIRMED]
- `Variable` defines `__hash__ = id` but every other node type is **unhashable**
  (`{u+v: 1}` raises), so expressions can't be used in sets/dict caches. [CONFIRMED]

A library-wide grep found no internal reliance on expression membership today, so this
is a user-facing trap plus a latent footgun for future internal code. **Fix:** define
`__hash__ = object.__hash__`-style identity hash and `__bool__` on `Constraint`
raising `TypeError("Constraint is not a boolean — did you mean to add it with
subject_to()?")` (the cvxpy approach). That turns both silent traps into immediate,
explained errors while keeping the DSL. Also give `Constraint` `eq=False` in the
dataclass decorator (identity comparison).

### M5. `subject_to` mutates the caller's `Constraint`; duplicate names unchecked

`subject_to(c, name=...)` assigns `c.name` in place; adding the same object under two
names leaves **both** model entries with the second name (confirmed: both entries are
literally the same object). `validate()` checks variable-name uniqueness but not
constraint names, while `SolveResult.constraint_duals` is **keyed by constraint name**
— silent dual-entry collisions. Fix: copy-on-name (`dataclasses.replace`) or reject
re-adding an already-added Constraint object; add duplicate-name validation (or
auto-suffix with a warning).

### M6. Misspelled `solve()` kwargs vanish

`m.solve(gap_tolerence=1e-3)` runs to completion at the default tolerance
[CONFIRMED]. The `**kwargs` funnel forwards to backend layers where unknown keys are
absorbed. For a solver whose options change numerical behavior, a typo'd option
silently not applying is a results-integrity issue (user believes they tightened the
gap). Fix: validate the union of accepted keys at the `solve_model` dispatch layer;
unknown keys raise `TypeError` listing near-matches. Requires a one-time inventory of
every backend's accepted kwargs (AMP, gurobi, mip-nlp, …), which is also useful
documentation.

### M8. Shape errors are not caught at build time

The modeling layer performs **no shape inference**: `a(3,) + b(2,) <= 1` builds
fine, and depending on path either dies later with a deep JAX traceback
(NLPEvaluator) or — worse — silently "solves" via M1's aggregation. After M1's fix,
add lightweight shape propagation to `Expression` construction (numpy broadcast rules
on a cached `.shape` attribute; raise on incompatibility at operator time with the
two operand shapes in the message). This is where cvxpy and JuMP are strictly ahead;
it converts a class of deep-compile errors into immediate one-line errors.

### M9–M12 (smaller, [BY INSPECTION])

- **M9.** `Disjunct.subject_to` accepts only `Constraint` or `list` — a generator is
  appended *as one item* and fails much later (attribute assignment on a generator)
  with a baffling error; `Model.subject_to` handles generators properly. Unify.
  Also `Disjunct.subject_to(…, name=…)` silently ignores its `name` parameter.
- **M10.** `resolve_indexed_values` (indexed.py): a `dict` spec with **extra keys**
  not in the set is silently accepted (misspelled member alongside correct ones is
  ignored). Warn/raise on unused keys.
- **M11.** `land()`/`lor()` skip the `_wrap_logical` validation their operator
  equivalents perform; `land(y1, 5)` builds a malformed tree that fails downstream.
- **M12. ✅ RESOLVED (X-1).** `num_constraints` counted `Constraint` *objects*, not
  scalar rows, and excluded `_builder_linear_blocks` rows — `summary()` under-reported
  fast-path models (a fast-API-only model reported `0`). **FIXED:** `num_constraints`
  now adds `Model._num_builder_constraint_rows()` (the builder blocks' scalar-row
  count); `Model._has_builder_only_rows()` was added as the shared predicate. Regression:
  `test_x1_builder_resident_rows.py::test_m12_*`. (The residual sub-item — counting
  *vectorized* expression `Constraint` objects as their flattened row count — is a
  cosmetic display nit tracked under modeling, separate from the fast-path correctness
  hazard closed here.)

---

## 4. Performance

- **P-M1. `_check_name` is O(n²)** (`core.py:3213`): rebuilds
  `{v.name for v in _variables} | {p.name ...}` on every declaration. Measured:
  37/94/170 µs per variable at n = 1k/2k/4k (pure quadratic growth; ~17 s of pure
  name-checking at 20k declarations). Fix: maintain a persistent `self._names: set`
  updated in `_register_variable`/`parameter`. One-line-class of change, bounded
  risk. (Note: `Model.constraint`'s fast linear-family path and
  `add_linear_constraints` are well designed — this fix removes the last O(n²) in
  the declaration path.)
- **P-M2. `_multiply_terms`/`sum` left-fold chains** build O(n)-deep trees for
  `prod()` and list-form sums. `SumOverExpression` already flattens sums; `prod`
  should get a matching n-ary node (or document the depth limit). Low priority.
- **P-M3.** After M1 stage 2, vectorized models regain the fast algebraic
  extraction; until then the correctness fix (stage 1) deliberately trades their
  speed for correctness — state this in the changelog.

---

## 5. Test coverage gaps

There is **no dedicated test file for `core.py`** — Model/Expression/Constraint
semantics are only exercised incidentally through solver tests. Specifically missing
(each maps to a finding):

1. Vector-constraint row-expansion contract across **all** solve paths (M1) — the
   nearest miss in the whole suite: scalar loops are tested everywhere, broadcast
   forms nowhere.
2. `.nl` round-trip preserving variable bounds for every var type, incl. fixed
   binaries (M2).
3. Cross-model expression rejection (M3).
4. `__eq__`/`__bool__`/hash behavior pinning (M4).
5. `subject_to` aliasing/mutation semantics and duplicate-name handling (M5).
6. `solve()` kwarg validation (M6).
7. Build-time shape errors (M8).
8. LaTeX: a golden-file test of a model exercising precedence corner cases
   (`-x**2`, nested fractions, un-normalized constraints) — current tests check
   substrings only.

---

## 6. Delegated-review findings: `gams_parser.py`, `examples.py`, `latex.py`, `implicit.py`

All findings below were verified by the delegated passes with runnable repro inputs
unless marked SUSPECTED; the two headline items (G1, E2) were independently
spot-checked in this review. Line numbers refer to the files as of this date.

### 6.1 `gams_parser.py` — systemic silent-failure design

The parser handles the happy-path MINLPLib subset, but its **default failure mode is
silent**: unknown characters, unmatched senses, unevaluable expressions/conditions,
unresolvable bound targets, and non-constant bound right-hand sides are dropped or
defaulted (usually to `0.0`/true) rather than raised. Each such drop yields a
syntactically valid but semantically different model that the solver then certifies.
Verified defect classes, worst first:

| ID | Loc | Finding (all VERIFIED unless noted) |
|----|-----|-------------------------------------|
| G1 | `:1578` | **`model m /eq1, eq2/` equation subsets ignored** — `_build_equations` iterates all `equation_defs`; the stored subset is never read. Excluded equations are added anyway → over-constrained model, false infeasible/optimum possible. Every test fixture uses `/all/`, so untested. |
| G2 | `:1663-1673, :1619-1631` | **Objective-defining equation eliminated even when the objective variable is used or bounded elsewhere** — `defz.. z =e= x; z.up = 4; maximize z` parses to `maximize x` (optimum 10 instead of 4); a second constraint on `z` is left dangling while the coupling is deleted. |
| G3 | `:2171-2173` | **Indexed bounds with non-literal RHS silently dropped** — `x.up(i) = b(i)` leaves `ub = 1e20` (evaluator has no `ExprIndex`/`ExprOrd` case and `None` → silent `continue`; the eval also sits *outside* the index loop so per-element values are impossible). Dropped upper bound = relaxed model. |
| G4 | `:700-712` | **Quoted element labels in domains discarded** — `x.fx('i1') = 2` parses to an empty domain and **fixes all elements** to 2. Tightened into a different model, no warning. |
| G5 | `:741-748` | **Comma-less parameter data corrupted** (the standard newline-separated GAMS style) — `/ i1 10 \n i2 20 /` parses as `{('i1','10','i2'): 20.0}`; all lookups then default to `0.0`. |
| G6 | `:770-829` | **Table parsing corrupts data in four common cases**: negative values (row scan stops at `-`; 3 of 4 cells lost), sparse tables (values assigned positionally, mis-keyed — GAMS tables are column-position-significant), numeric column headers (skipped), composite row labels `i1.j1` (mis-keyed). |
| G7 | `:138-139` | **Lexer silently deletes unknown characters** — `z =e= x^2` parses as `z == x`; also eats `%`, `&`, `#`/`!` comments. Should be a loud lex error. |
| G8 | `:977-1134` | **Term-level `$` conditions inside equation bodies dropped** — `z =e= y$(x > 0)` emits `z == y` unconditionally (only a generic "unrecognized statement" warning for the leftovers). |
| G9 | `:1997, :2038, :1606, :2145, :2195` | **Un-evaluable `$` conditions treated as TRUE** — `sum(i$myset(i), …)` with undefined `myset` sums everything. `None` (couldn't evaluate) must be a loud error, not "include". |
| G10 | `:2172, :692` | **`INF`/`EPS`/`NA` unsupported** — `x.lo = -inf` silently skipped (positive variable stays at 0); `scalar bigM / inf /` crashes with a token error on valid GAMS. |
| G11 | `:2159` | **Case-sensitivity: GAMS is case-insensitive; the bound path is case-sensitive and silently `continue`s** — `variables X; x.lo = 5;` drops the bound (and hides genuine typos). |
| G12 | `:1513, :1531` | **Integer default `ub=1e6` hardcoded** (GAMS default is +inf) and `n.up = inf` can't undo it because of G10. |
| G13 | `:1030-1036` | **`**` parsed right-associative; GAMS is left-to-right** — `2**3**2` = 512 here, 64 in GAMS. |
| G14 | `:618-636` | **Quoted set elements dropped** — `set i /'a','b'/` → empty set; downstream sums vanish. |
| G15 | `:482-490, :1283-1288` | **`if(cond, body)` crashes or executes the body unconditionally** (the code comments the unconditional behavior). |

Also traced (SUSPECTED, loud-else class): `_add_constraint` has no `else: raise` on
unknown senses; `_ensure_expr` maps unresolvable strings to `Constant(0.0)`;
`ExprOrd` falls back to `Constant(1.0)`; `smin`/`smax` discard their index lists;
dollar-control lines (`$include`!) skipped with no warning while `$ontext…$offtext`
bodies are *parsed as statements* (verified crash on an ordinary commented file); only
`solves[0]` is used but equations from the whole file are included; relational
operators (`<=`, `and`, `lt`, …) unsupported in `$` conditions (loud, but blocks many
real files); division-by-zero in constant eval returns `None` feeding G9; element→
position fallback searches *all* sets (wrong position possible with shared labels).

Performance (all measured or traced): `_element_index` does a linear `list.index`
per term → **O(n²) sum building** (0.14 s/0.54 s/2.03 s at n = 3k/6k/12k);
`_is_param_assign` rebuilds lowercased key-sets per statement; `_execute_body`
re-parses the loop body per index combination; `_apply_bounds` reallocates the full
bound arrays per element.

Test coverage: `test_gams.py` (749 lines) covers happy paths only — **every verified
bug above is untested** (all fixtures use `model /all/`, dense positive alphabetic
tables, comma'd data, unquoted labels, lowercase-consistent names).

### 6.2 `examples.py` — the gallery contains wrong and broken models

> **✅ RESOLVED E1, E2, E3** — `python/tests/test_examples_gallery.py` (this PR).
> - **E1**: reformulated `example_pooling_haverly` with the pool **concentration**
>   `p` (not sulfur mass) so the product specs bind even at zero pool flow. The
>   review's suggested "scale `2z` by `(y0+y1)`" was itself wrong — multiplying the
>   spec through by the pool flow makes it vacuous (`0 ≤ 0`) when the pool is
>   bypassed, and the model then certified 500 shipping 2 %-sulfur product against
>   the 1.5 % spec. The concentration form certifies the classic Haverly-I optimum
>   **400** with the product-1 spec binding at exactly 1.5 %.
> - **E2**: `example_logical_constraints` now uses `m.logical()` (not `subject_to`)
>   for the propositional constraints, `m.make_disjunct()` (not `m.disjunct`), and
>   encodes "project 3 requires project 0" as the implication `lor(~a3, a0)` (not the
>   `land` conjunction that forced both). Also fixed `Disjunct`'s docstring
>   (`Model.make_disjunct`). Builds and `validate()`s.
> - **E3**: `example_reactor_design`'s heat balance dropped the needless `F/(Cp·F)`
>   and reduced the exotherm so the adiabatic cascade stays within the 750 K limit
>   (`T=[400,520,640]`); the model is now feasible (was provably infeasible). The
>   `test_nlp_ipopt.py` "hard example" carve-out that accepted `INFEASIBLE` is
>   removed — reactor takes the real `OPTIMAL/ITERATION_LIMIT` check.
>
> A whole-gallery smoke test now builds and `validate()`s every pure-modeling
> `example_*`. Verified fails-before/passes-after. The full finding text is below.

| ID | Loc | Finding |
|----|-----|---------|
| E1 | `:88-91` | **Haverly pooling model mathematically wrong** [VERIFIED]: `product1_sulfur_spec` clears the `1/(y0+y1)` denominator on two of three terms but leaves `2z` unscaled (dimensionally inconsistent). Solved "globally optimal" objective **1390** with product-1 sulfur **2.49 %** against the 1.5 % spec the constraint claims to encode; the classic Haverly-I optimum is **400**. Also `p`'s ub 300 cuts feasible pool states (max sulfur mass is 400). Every test touching this example asserts only "converges/finite". |
| E2 | `:431-448` | **`example_logical_constraints` crashes** [VERIFIED + spot-checked]: `m.subject_to(dm.atmost(...))` — `subject_to` doesn't accept `LogicalExpression` (the API is `m.logical`); `m.disjunct(...)` doesn't exist (it's `make_disjunct` — note `Disjunct`'s own docstring also says "Created via Model.disjunct", so docstring and gallery agree with each other and both are wrong). Plus a logic bug: "project 3 requires project 0" encoded as `land(~a3, a0)` forces both values unconditionally; it must be the implication `lor(~a3, a0)`. Shipped crashing because the `__main__` harness swallows exceptions and no test builds it. |
| E3 | `:276-279` | **`example_reactor_design` is provably infeasible** [VERIFIED by interval arithmetic]: the heat balance simplifies to `T[i] = T[i-1] + 320` (the `F/F` also cancels — a needless variable division), so `T[2] ≥ 940` against ub 800 and `max_temperature ≤ 750`. `test_nlp_ipopt.py:128-132` quarantines it as a "hard example" with a loosened assertion — a workaround masking global infeasibility, exactly the pattern the development philosophy forbids. |

### 6.3 `latex.py`

> **✅ RESOLVED L1, L5, L6, L8** — `python/tests/test_display_cluster.py` (display-cluster PR).
> - **L1**: `_constraint_to_latex` now renders a `\text{[Type: name]}` placeholder
>   for non-arithmetic constraint types (indicator/disjunctive/SOS/logical) instead
>   of raising `AttributeError` — GDP models display without crashing.
> - **L5**: indexed digit-suffixed variables merge into a single subscript
>   (`y1[0]` → `y_{1,0}`, not the invalid `y_{1}_{0}`); other bases are brace-wrapped.
> - **L6**: `_fmt_num` guards non-finite input (`\infty`/`-\infty`/`\mathrm{nan}`)
>   instead of `int(inf)` raising `OverflowError`.
> - **L8**: `Expression._repr_latex_` now renders the DAG via `expr_to_latex`.
>
> Verified fails-before/passes-after. **Still open** (this PR did not touch them):
> L2 (fast-path rows invisible — part of the X-1 builder-rows root), L3
> (`SumOverExpression`/`Parameter`), L4 (precedence on non-BinaryOp nodes), L6's
> slice rendering, L7 (HTML vs LaTeX escaping).

| ID | Loc | Finding |
|----|-----|---------|
| L1 | `:152-161` — ✅ RESOLVED | **`to_latex`/`_repr_latex_`/`_repr_html_` crash on any model containing `if_then`/`either_or`/`m.logical` constraints** [VERIFIED]: `_constraint_to_latex` reads `.sense`/`.body` on `_IndicatorConstraint` et al. Merely *displaying* such a model in Jupyter raises `AttributeError`. |
| L2 | `:200-209` | **Fast-path constraint families are invisible** [VERIFIED]: renderer reads only `_constraints`, so a `m.constraint(...)` model renders "(1 variable, 0 constraints)" — a display that misrepresents the model. |
| L3 | `:26-37` | **`SumOverExpression` and `Parameter` unhandled** [VERIFIED]: the dominant `dm.sum(..., over=...)` pattern renders as literal `Σ[6 terms]`; `Parameter` as `param(price_A)` with a bare `_` in math mode. |
| L4 | `:109-127` | **Precedence bugs on non-BinaryOp nodes** [VERIFIED]: `(-x)**2` → `-x^{2}`; `dm.sum(x)**2` → `\sum x^{2}`; `(A@x)**2` → `A\,x^{2}` — all mathematically wrong renderings (UnaryOp/Sum/MatMul ignore `parent_prec`). |
| L5 | `:103-106` | **Invalid LaTeX double subscript** [VERIFIED]: variable `y1` indexed → `y_{1}_{0}` (MathJax error box). |
| L6 | `:103-105, :61-65` | Slices render as `slice(None, None, None)`; `_fmt_num(inf)` raises `OverflowError` (reachable bare via `_constraint_to_latex`) [VERIFIED]. |
| L7 | `:251-256` | `_escape_text` conflates HTML and LaTeX escaping regimes (inserts `&amp;` into `aligned` math; leaves `%`, `_`, `#` unescaped) [SUSPECTED]. |
| L8 | `core.py:173-175` | `Expression._repr_latex_` wraps the plain repr in `$...$` instead of calling `expr_to_latex` — bare expressions display as broken math in Jupyter [BY INSPECTION]. |

### 6.4 `implicit.py`

| ID | Loc | Finding |
|----|-----|---------|
| I1 | `:63-77` | **Absolute-only convergence tolerance** [VERIFIED]: a residual scaled by 1e8 converges to machine-precision `v=√2` but fails the `rnorm ≤ 1e-10` gate → NaN poisons the whole NLP solve. Loud (per house philosophy) but a false refusal on well-posed input; add a relative/step-size criterion. |
| I2 | `:146-150` | Build-time probe calls `residual(zeros(len(u_inputs)), x0)` but runtime flattens/concatenates vector inputs, so the probe's `u` length is wrong for any non-scalar input — the shape check can silently pass or false-fail [SUSPECTED, partially verified]. Also `tol<=0`/`max_iter<1` unvalidated (`max_iter=0` → every evaluation NaN with no build-time complaint). |

Also noted: the docstring's higher-order-AD claim was spot-verified correct
(`jax.hessian` through the node gives the exact −1/32 at u=4) but is pinned by no test.

---

## 7. SOTA assessment — the modeling API against Pyomo, JuMP, gurobipy, cvxpy, GAMSpy

### 7.1 Where it stands

The API is a **hybrid of the two modern modeling styles** — numpy-vectorized
operator overloading (gurobipy matrix API / cvxpy style: `A @ x <= b`, broadcast
arithmetic, shaped variables) *and* named-set indexed modeling (Pyomo/GAMS style:
`m.continuous("ship", over=plants*markets)`, rule-based constraint families with
`Skip`). Few tools offer both; the set layer's fast path that lowers affine
constraint families straight into the Rust arena (`_try_fast_linear_family` →
`add_linear_constraints`) is a genuinely good design that addresses Pyomo's
best-known weakness (Python-object model-build overhead) without giving up the
symbolic layer.

Features that are *ahead* of the mainstream open-source baseline:

- **GDP as a first-class citizen**: disjunctions, `if_then`, boolean propositional
  logic with lowering, `Disjunct` blocks, and `complementarity()` with GDP/SOS1
  reformulations — Pyomo.GDP-class functionality that JuMP/gurobipy lack natively.
- **`if_else` with certified semantics**: hull-forced disjunction plus interval-
  derived sound bounds on the auxiliary — a *global-solver-aware* conditional that
  none of the local-solver APIs can offer honestly.
- **A three-tier user-function ladder with explicit certificate semantics**:
  `udf` (symbolic, full global support) → `custom` (opaque JAX-AD, local-only,
  integers rejected, `gap_certified=False`) → `implicit` (IFT-differentiated inner
  solve with nonsingularity/convergence gates). The *documented downgrade of the
  optimality certificate* when opacity is introduced is exactly right and rare —
  JuMP's `register()` has no equivalent story.
- **Parametric sensitivities** (`Parameter` + envelope-theorem `result.gradient()`)
  built into the core result object — differentiable-optimization territory
  (cvpylayers-adjacent), unusual for an MINLP tool.
- Ergonomics: PSE-form LaTeX/HTML reprs, `compute_iis()`, LLM formulation hooks,
  `.nl`/GAMS/MPS/LP import-export breadth.

### 7.2 Where it falls short of SOTA

1. **Expression-time diagnostics.** cvxpy and JuMP validate shapes, ownership, and
   argument types *at operator time* with precise errors. Here, shape mismatches
   (M8), cross-model mixing (M3), and boolean-context misuse (M4) all pass silently
   and surface — if at all — as deep JAX tracebacks or wrong answers. This is the
   single biggest maturity gap, and (per M1) it is not just ergonomics: silent
   contract violations become silent wrong certificates downstream.
2. **Model lifecycle.** No constraint removal/modification, no model copy, no
   fix/unfix variable API, no warm modeling loop (`Parameter` covers value changes
   only, not structure). JuMP/gurobipy treat incremental modification as core.
3. **Solver-options hygiene** (M6): every mainstream API errors on unknown options.
4. **Missing constructs** vs. the big APIs: semi-continuous variables, SOS weights
   (`sos1(vars)` takes no weight vector), general integer→binary expansion helpers,
   piecewise-linear helper (`pwl`) — pyomo/gurobipy have all four; the GDP layer can
   express PWL but a dedicated helper with SOS2/logarithmic formulations is standard.
5. **Vectorized nonlinear reductions**: `dm.sum(x, axis=…)` exists but there is no
   shaped-expression introspection (`expr.shape`), which blocks users from writing
   shape-generic model libraries — and is prerequisite to M8's fix anyway.

### 7.3 Verdict

Design-wise this is a **modern, well-conceived modeling layer** — the numpy+sets
hybrid, the Rust fast path, and the certificate-aware user-function ladder are at or
beyond the state of the art for open-source MINLP front-ends. Implementation-wise it
is let down by missing *guardrails* (M3/M4/M6/M8) and by two outright P0s (M1's
vector-row aggregation on the LP/MILP path, M2's binary-bound loss) that break the
solver's central promise on models written exactly the way the API's own docstrings
teach. The guardrail work is mechanical; the payoff (silent-wrong-answer classes
converted to immediate errors) is disproportionate.

Two peripheral components lag well behind that core: the **GAMS importer** is far
from parity with real-world GAMS (GAMSpy / the GAMS Connect ecosystem is the
reference bar) and, worse, fails *silently* by design (§6.1) — for a
certificate-producing solver an importer must be either correct or loud, never
quietly approximate; and the **examples gallery** ships models that are wrong,
crashing, or infeasible (§6.2), which for an API that doubles as the LLM-formulation
tool schema is training data for mistakes. Both are fixable with the phased plan
below; neither taints the core API design.

---

## 8. Implementation plan (for Opus)

House rules per CLAUDE.md: feature branch + PR per phase, task ID in title, every fix
carries a regression test that fails before / passes after, run the modeling-adjacent
suites (`test_sets*`, `test_indexed_*`, `test_model_latex`, `test_implicit_node`,
`test_gams*`) plus `pytest -m smoke` and the adversarial suite; state results in the
PR. Baseline recorded above. M1 touches the solver's classification layer: its fix is
*bound-relevant* — run the differential checks specified below, not just unit tests.

### Phase 1 — P0s (PR `fix(correctness): M-1..M-3`)

| ID | Task | Files | Acceptance criteria |
|----|------|-------|---------------------|
| M-1a | Algebraic extractors: raise `_NotLinearError`/`_NotQuadraticError` on any array-shaped node in scalar position (delete the array-as-sum branches) | `_jax/problem_classifier.py:219-226, 372-376` | LP/vector-eq/MILP repros (§2 M1) return true optima with elementwise-feasible points; scalar-loop controls unchanged; full benchmark smoke suite `incorrect_count ≤ 0`; property test: algebraic vs autodiff extraction agree exactly on 100 random affine models (scalar bodies) |
| M-1b | Row-expanding algebraic extraction for array bodies (shape propagation through `+/-/*` and broadcasting) to restore the fast path | same | Extracted `(A,b,c)` bit-identical to autodiff extractor on a vectorized-model panel incl. DAE-style blocks; build-time measurement vs M-1a fallback included in PR |
| M-2 | `from_nl`: apply parsed bounds to binary variables (intersect with `[0,1]`) | `core.py:3627-3632` | Round-trip repro preserves optimum 11.0; hand-written `.nl` with fixed binary column imports fixed; `from_pyomo` round-trip test with a fixed binary |
| M-3 | Ownership validation in `Model.validate()`: every Variable/Parameter in objective+constraints must satisfy `node.model is self` | `core.py` (validate; reuse `_find_owning_model`-style walk, extended to CustomCall args and Parameter) | Cross-model repro raises `ValueError` naming `xb` and model `B`; solve-path overhead measured (one DAG walk) |

### Phase 2 — guardrails (PR `fix(modeling): M-4..M-8`)

| ID | Task |
|----|------|
| M-4 | `Constraint.__bool__` → `TypeError` with guidance; identity `__hash__` on `Expression`; `eq=False` on the `Constraint` dataclass. Sweep the codebase for any `bool(constraint)`/membership reliance first (none found in this review, but re-verify at implementation time) |
| M-5 | `subject_to`: copy-on-rename (`dataclasses.replace`) instead of in-place mutation; reject double-adding the same object; validate/auto-suffix duplicate constraint names (duals are keyed by name) |
| M-6 | Strict kwarg validation at the `solve_model` dispatch layer: inventory per-backend accepted keys; unknown → `TypeError` with near-miss suggestion |
| M-8 | Shape propagation on `Expression` (cached `.shape`, numpy broadcast rules; raise at operator time on mismatch). Prerequisite for and verified together with M-1b |

### Phase 3 — polish (PR `fix(modeling): M-9..M-12` + perf)

M-9 (Disjunct.subject_to generator/name handling), M-10 (unused-dict-key warning in
`resolve_indexed_values`), M-11 (`land`/`lor` validation), M-12 (row-accurate
`num_constraints`/`summary`), P-M1 (persistent name set — O(n²) → O(n) declarations;
include the µs/var measurement before/after), P-M2 (n-ary `prod` node) — each small,
each with its one regression test.

### Phase 4 — SOTA-gap features (design first, separate PRs)

In value order for this repo: (1) model-modification API (remove/replace constraint,
fix/unfix variable) — unlocks resolve loops and OBBT-style experiments at the Python
layer; (2) piecewise-linear helper with SOS2/logarithmic/GDP formulations (reuses the
existing GDP machinery; feeds the global solver well-structured relaxations); (3) SOS
weights + semi-continuous variables; (4) `expr.shape` public API on top of M-8. Each
needs an entry-experiment/kill-criterion note per the development philosophy before
implementation.

### Phase 5 — examples gallery (PR `fix(examples): M-E1..M-E3`)

| ID | Task | Acceptance criteria |
|----|------|---------------------|
| M-E1 | Fix Haverly pooling: scale the `2z` term by `(y0+y1)` (or model the spec with the explicit fractional form) and correct `p`'s ub | Solve certifies the classic Haverly-I optimum **400**; new test pins it (currently no test pins any gallery optimum except `simple_minlp`) |
| M-E2 | Fix `example_logical_constraints`: `m.logical(...)` instead of `subject_to`, implication instead of conjunction, `make_disjunct`; align `Disjunct`'s docstring with the real method name; make the `__main__` harness fail loudly | Example builds and solves; smoke test constructs **every** `example_*` function and asserts `validate()` passes |
| M-E3 | Fix `example_reactor_design`'s heat balance (sign/magnitude of the `dH` term; remove `F/F`) so the model is feasible and matches its docstring; remove the "hard example" carve-out in `test_nlp_ipopt.py` | Model solves feasible; the loosened assertion is restored to a real check |

### Phase 6 — GAMS parser hardening (PR series `fix(gams): M-G1..`; largest work item)

Ordering principle: first make silent failures **loud** (cheap, converts every
remaining unknown-unknown into a visible error), then fix semantics, then perf.

1. **Loud-failure sweep (M-G0):** lexer errors on unknown characters (G7); `else:
   raise` on unknown senses (B-class); un-evaluable `$` conditions and bound RHS
   evaluations raise instead of include/skip (G3, G9, G10, G11); `$include`/dollar-
   control emit at least a warning; `$ontext…$offtext` actually skipped. This single
   PR removes the parser's systemic silent-default design.
2. **Model-semantics fixes:** equation subsets honored (G1); objective elimination
   gated on "objective variable otherwise unused and unbounded", else fall back to
   the sound MINLPLib path (G2); per-element indexed bound evaluation inside the
   loop with `ExprIndex`/`ExprOrd` support (G3); quoted labels in domains and set
   elements (G4, G14); comma-less parameter data (G5); table parser rewritten
   column-position-aware with negative-number, sparse, numeric-header, composite-key
   support (G6); term-level `$` (G8 — parse-and-raise is acceptable initially);
   case-insensitive symbol tables (G11); integer default bound = +inf with `INF`
   literal support (G10, G12); left-associative `**` (G13); `if` statements honored
   or refused loudly (G15).
3. **Perf:** per-set `{elem: pos}` maps (kills the measured O(n²) sum build),
   cache `_is_param_assign` key-sets, pre-parse loop bodies once, in-place bound
   array updates.
4. **Test corpus:** add adversarial fixtures for every G-item (each must fail
   before/pass after), plus a differential harness: parse a curated set of real
   MINLPLib `.gms` files and compare optima against `minlplib.solu` — the oracle
   the repo already ships (`~/Dropbox/projects/discopt-minlp-benchmark/`).

### Phase 7 — rendering & implicit polish (PR `fix(modeling): M-L1..M-I2`)

L1 (guard non-`Constraint` entries — render `if_then`/disjunctions/logical rows
symbolically or as a summary line; a *display* must never crash a model),
L2 (render `_builder_linear_blocks` rows or an explicit "+N fast-path constraints"
summary — silently hiding constraints misrepresents the model), L3 (add
`SumOverExpression`/`Parameter` cases), L4 (thread `parent_prec` through
UnaryOp/Sum/MatMul), L5–L7 (brace subscript bases, slice rendering, `isfinite` guard,
split HTML vs LaTeX escaping), L8 (`Expression._repr_latex_` → `expr_to_latex`),
I1 (relative + step-size convergence criterion), I2 (probe with flattened `u` size;
validate `tol`/`max_iter`; pin the Hessian-through-the-node test). Golden-file LaTeX
test covering: a GDP model, an indexed-sum model, a fast-path model, `y1[0]`,
`(-x)**2`, `dm.sum(x)**2`.
