# Solver Core — Algebraic Extraction Review (the recurring P0)

**Date:** 2026-07-03
**Scope:** `python/discopt/_jax/problem_classifier.py` — specifically the algebraic
LP/QP extraction path (`_extract_linear_coefficients`,
`_extract_quadratic_coefficients`, `_extract_constraints_algebraic`,
`extract_lp_data`, `extract_qp_data`) that classifies and lowers models on the
fast solve path. This is a **targeted** review of the extraction layer three
prior module reviews already pointed at (modeling-review M1, ro-review ADJ-1),
not a review of the whole 55k-line `_jax/` tree.
**Method:** Traced the walkers by hand and reproduced every claim end-to-end
against the installed package.

**This is the single highest-impact correctness cluster in the review series that
produces wrong *certified answers* (as opposed to the llm RCE, which is a security
issue).** It fires on models written exactly the way the modeling API's own
docstrings teach — numpy-vectorized `A @ x`, `dm.sum(c * x)`, broadcast `a + b` —
and returns a confidently-wrong optimum with `status="optimal"`.

---

## 1. The two confirmed P0s (one root cause)

Both stem from the algebraic extractor assuming every constraint/objective body is
a **single scalar** and every array node can be **summed away**.

### CORE-1 — Vector constraints collapse to one summed row (= modeling M1)

`_extract_constraints_algebraic` (`:614`) does `a_row, const =
_extract_linear_coefficients(con.body, model, n_orig)` and appends **one** row per
`Constraint` object. A vector body — `a + b <= 1` with `a,b` shape `(2,)` — is one
`Constraint` whose body is array-valued, so it becomes **one** row. Inside the
walker, the bare-array-variable branch (`:224`) literally comments *"Array variable
treated as sum when used as scalar"* and sums every element into a single
coefficient. Net: `a + b <= 1` is extracted as `Σa + Σb <= 1`.

**Reproduced** (`status="optimal"` in every case):
- LP `min Σa+Σb s.t. a+b>=1` (shape (2,), box [0,1]) → objective **1.0**, true
  **2.0**; the returned point is **infeasible** by the model's own evaluator.
- Vector **equality** `a+b==1` → same wrong 1.0.
- MILP set-cover `min Σy+Σz s.t. y+z>=1` (shape (3,) binaries) → **1.0**, true
  **3.0**.
- Scalar-loop form of the identical model → **correct** on the same path.

### CORE-2 — Maximize sense lost on `sum(const·vec)` bodies (= ro ADJ-1)

`_extract_linear_coefficients` handles `dm.sum(c * x)` by recursing through
`SumExpression` → `BinaryOp("*")` → `_eval_const(Const([1,1]))` at `:259`, which
calls `float(v.item())` on a **size-2 array** and raises a raw **`ValueError`**
(`:569`) — *not* `_NotLinearError`. The algebraic path aborts, falls over to a
different extractor, and that fallback **does not apply the maximize negation**:
`extract_lp_data` returns `c = [1, 1, 0]` for a *maximize* model, so the solver
minimizes `x₁+x₂` and returns **0**.

**Reproduced:** `maximize dm.sum([1,1]·x) s.t. dm.sum([1,1]·x) <= 4` →
objective **1.5e-08** at `x≈[0,0]`, true **4.0**. `extract_lp_data(m).c` is
`[1,1,0]` (un-negated). Scalar / `x[0]+x[1]` / `A@x` / `dm.sum(x)` forms all solve
correctly — only the `sum(const_vec * var)` body shape fails.

Two defects compound here: (a) raising `ValueError` instead of `_NotLinearError`
means the intended "not linear → use autodiff" routing is bypassed by an *exception
type mismatch*; (b) whichever fallback catches it drops the objective sense. Either
one alone would be a bug; together they yield a wrong certified optimum.

### The quadratic path has the same latent defect

`_extract_quadratic_coefficients` (`:323`) carries an identical "array variable
treated as sum" branch (`:372` region). End-to-end QP probes happened to come out
correct in the ro review (the asymmetric instance routed elsewhere), but the branch
is present and must be fixed/audited identically — a vectorized QP objective or
constraint will hit it.

---

## 2. Why this survived, and why it matters

- **The DAE module escapes only by accident**: its vector equalities are built from
  `MatMulExpression` bodies, which raise `_NotLinearError` at `:304/:314` and route
  to the (correct) autodiff extractor. Change a DAE body to a broadcast form and it
  would fall into CORE-1.
- **No test exercises a vectorized LP/MILP body** — the entire suite writes
  constraints as scalar loops, which is exactly the path that works. The nearest
  miss in the whole codebase.
- For a solver whose product is a *certificate*, this is the worst failure mode:
  `incorrect_count ≤ 0` is the repo's hard gate, and these models violate it
  silently on the default `solve()`.

---

## 3. Fix

**Stage 1 (minimal, provably safe — do this first):** in both
`_extract_linear_coefficients` and `_extract_quadratic_coefficients`, **raise
`_NotLinearError` on any array-shaped node in scalar position** — bare array
`Variable`, array `Constant`, and any broadcasted intermediate — and make
`_eval_const` raise `_NotLinearError` (not `ValueError`) on a non-scalar. Delete the
"treated as sum" branches. Every affected model then routes to
`_extract_lp_data_autodiff` / the autodiff QP path, which are **documented-correct
for vector bodies** (one row per component, sense handled). This trades the fast
path's speed on vectorized models for correctness — acceptable and reversible.

Audit, in the same PR, every `except _NotLinearError` / `except Exception` around
the extractors to confirm the fallback they route to applies objective sense and
expands rows correctly (CORE-2 shows at least one fallback does not) — add a
regression asserting `extract_lp_data(maximize_model).c` is negated.

**Stage 2 (restore the fast path):** teach the algebraic extractor to expand array
bodies into proper per-element rows (shape inference through `+`, `-`, scalar `*`,
broadcast, `MatMul`) so vectorized models regain fast extraction. Verify
**bound-neutrally**: extracted `(A, b, c)` must equal the autodiff extractor's
output exactly on a panel of vectorized LP/MILP/QP models (including a maximize
case and a DAE-style block).

**Regression tests (fail before / pass after):** the three CORE-1 repros (LP,
vector-equality, MILP) asserting *both* objective value and elementwise feasibility
of the returned point; the CORE-2 maximize repro asserting objective 4.0 and
negated `c`; a property test comparing algebraic-vs-autodiff extraction (rows,
matrix, objective, sense) on randomly generated vectorized affine models; the
`_eval_const` non-scalar case asserting `_NotLinearError` (not `ValueError`).

---

## 4. Scope note: the rest of the solver core is unreviewed

This document covers **only** the algebraic extraction path. The following remain
without a dedicated review and each is a multi-session effort in its own right —
they should not be assumed sound on the strength of the module reviews done so far:

- **`solver.py`** (11,725 lines) — the orchestrator: solve dispatch, the B&B loop,
  path routing, dual recovery (`:8385`+), root certification instrumentation. The
  GP auto-path and the examiner's dual conventions were touched tangentially in the
  gp/validation reviews; the B&B correctness core was not.
- **`_jax/` relaxation layer** (~55k lines) — McCormick/factorable relaxations,
  convexity certification, FBBT, cutting planes, RLT, the NLP evaluators. This is
  where global-optimality *soundness* actually lives (a loose relaxation that is
  still valid only costs speed; an **unsound** relaxation cuts the optimum). It
  needs its own review regime per CLAUDE.md §5 (differential bound tests), not a
  read-through.
- **`solvers/`** (9,178 lines) — POUNCE LP/IPM, cyipopt, Gurobi, OA/ECP, native-AD
  NLP backends.
- **Rust `crates/`** — in-house simplex, B&B tree, `.nl` parser, FBBT/presolve.
  Cargo workspace; needs `cargo test -p discopt-core` plus differential checks, a
  separate effort from the Python reviews.

The recurring **builder-resident-rows blind spot** (modeling M12, gp GP-1, export
EX-2, validation VAL-1) also has one of its roots here: `classify_problem` /
`extract_*` read only `model._constraints`, never `model._builder_linear_blocks`.
The shared `iter_all_rows(model)` / `_has_builder_only_rows(model)` primitive
recommended across those reviews should land on `Model` and be consumed here too,
so classification, extraction, export, and validation share one definition of "the
model's rows."

---

## 5. Priority

Fix CORE-1/CORE-2 **first among all correctness findings in the series** (the llm
RCE is more severe overall but is a security fix on an optional path; this is a
wrong-certificate bug on the default path for idiomatic models). Stage 1 is a small,
safe, reversible change with an outsized correctness payoff.
