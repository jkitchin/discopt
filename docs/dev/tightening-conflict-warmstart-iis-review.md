# tightening / conflict / warm_start / infeasibility Review

**Date:** 2026-07-03
**Scope:** `python/discopt/tightening.py` (126), `conflict.py` (166),
`warm_start.py` (236), `infeasibility.py` (256), and the Rust FBBT engine they
depend on (`crates/discopt-core/src/presolve/fbbt.rs`).
**Method:** Delegated verification with numerical repros; the P0 independently
re-confirmed here (both manifestations).

Headline: a **P0 soundness bug in FBBT bound tightening** that both **cuts feasible
points** and **reports feasible models as infeasible** — and it chains into
`conflict.py` (invalid no-good cuts) and the certified relaxation path.
`warm_start.py` and `infeasibility.py` are verified correct.

---

## 1. Summary

| # | Severity | Component | Finding |
|---|----------|-----------|---------|
| TG-1 | **P0 soundness** | `tightening.py:116-120` + `crates/…/fbbt.rs:1204-1208` | FBBT collapses each array-variable **block to element-0's bounds**, then stamps that interval onto **every** element. On heterogeneous per-element bounds it **cuts feasible points** and can report a **feasible model as infeasible** [CONFIRMED, both manifestations re-run] |
| CF-1 | **P0 (chained)** | `conflict.py:81-85` | Uses `fbbt_box(...).infeasible` as its infeasibility oracle, so TG-1 makes it emit **invalid no-good cuts** that remove feasible assignments [CONFIRMED end-to-end] |
| MR-1 | P1 (suspected) | `_jax/milp_relaxation.py:4338-4380` | `_fbbt_argument_box` consumes the same `fbbt_box` result to build univariate relaxation envelopes on the rescue path; trusting an unsound box means the envelope can cut feasible points in a real certified solve [SUSPECTED — trigger-specific, not full-solve-reproduced, but the trusted input is provably unsound] |

Verified **correct** (with evidence):

- **`warm_start.py`** — rejects non-`Variable` keys (`TypeError`) and foreign-model
  variables (`ValueError`); clamping + integer rounding land in-bounds and integral
  with warnings; and crucially it **never certifies** — it returns only the flat
  start vector, sets no incumbent/bound, so it cannot corrupt the certificate. (Two
  non-bugs noted: it relies on CPython dict identity because `Variable.__eq__`
  returns a `Constraint` — a latent trap for any future refactor comparing distinct
  Variable instances by `==`, same root as modeling-review M4; and round-then-clip
  can yield a non-integer start for an integer var with *non-integer* bounds —
  harmless.)
- **`infeasibility.py`** (IIS) — refuses on feasible/inconclusive baselines
  (`ValueError`), drops a member only on a proven `"infeasible"` status, and its
  deletion filter is a valid single pass yielding a genuinely irreducible IIS
  (verified: 2-member IIS with `proven_irreducible=True` on `y≤2 ∧ y≥5`); binaries
  correctly excluded from bound candidates; snapshot/restore consistent. Its own
  logic is sound — it only inherits risk *if* the solver's presolve rides TG-1.

---

## 2. TG-1 in detail (the P0)

The Rust FBBT engine treats each variable **block** as a single scalar: it seeds
each variable's interval from element 0's bounds (`fbbt.rs:1204-1208`,
`v.lb.first()/v.ub.first()`) and returns one `Interval` per block. `tightening.py`
then applies that single interval to **every** scalar slot of the block
(`:116-120`), taking `max`/`min` against each element's original bound. When the
elements have *different* bounds, element 0's (tighter) bound is illegally
propagated onto the others. The module's own soundness claim — "a valid outer bound
for every element of the block" (`:108-110`) — is false: element 0's bound is not
an outer bound for the other elements.

**Reproduced (both re-run in this review):**
- **Cuts feasible points:** `x = continuous(shape=(2,), lb=[8,0], ub=[10,10])` →
  `fbbt_box` returns `lb=[8,8]`, `n_tightened=1`. Element `x[1]`'s lower bound
  jumped **0 → 8**, deleting the feasible region `x[1] ∈ [0,8)`.
- **False-infeasible on a feasible model:** `x = continuous(shape=(2,), lb=[5,0],
  ub=[5,3])` → `fbbt_box` returns `lb=[5,5], ub=[5,3]`, **`infeasible=True`**. The
  point `x=[5, 0..3]` is plainly feasible; element 1 got element 0's `lb=5`
  against its own `ub=3` → `lb>ub` → false infeasible. This directly violates
  "FBBT never reports a feasible problem as infeasible" — a false certificate.

**Why CI is blind:** `test_tightening.py:68`
(`test_never_loosens_and_handles_vector_vars`) is the only array test and uses
**homogeneous** bounds (`lb=0, ub=10` for all elements) — exactly the case where
the collapse-to-element-0 is invisible.

**Fix:** make the FBBT path element-aware for array blocks — either have the Rust
engine carry/return per-scalar intervals, or, as an interim Python guard, only
apply a block interval to elements whose *original* bounds equal element 0's, and
never let a block-level tightening narrow a distinct element (and never derive
`infeasible` from a collapsed block). **Regression test:** the two repros above —
the over-tighten must leave `x[1].lb=0`, and the feasible model must not be
reported infeasible.

## 3. CF-1 (chained)

`conflict.py`'s entire soundness argument rests on FBBT never false-reporting
infeasible. **Reproduced end-to-end:** a model with `b = binary()` and
`x = continuous(shape=(2,), lb=[5,0], ub=[5,3])` — feasible for both `b=0` and
`b=1` — makes `find_conflict_cuts(m, max_order=1)` return **two** cuts,
`(1−b) ≤ 0` and `b ≤ 0`, which together make `b` infeasible. Any cut here is
invalid; `add_conflict_cuts` would inject them and cut the true optimum. The
no-good construction and minimality logic are themselves correct — the invalidity
is entirely inherited from the FBBT oracle, so **fixing TG-1 fixes CF-1.**

## 4. Plan (for Opus)

**Phase 1 — `fix(correctness): TG-1` (Rust + Python).** Element-aware FBBT for
array blocks (per-scalar intervals from the engine, or the Python interim guard);
never collapse to element 0; never derive `infeasible` from a collapsed block.
Acceptance: both §2 repros fixed (no feasible cut, no false infeasible); CF-1's
end-to-end repro yields **zero** cuts on the feasible model; the homogeneous-bounds
test still passes; `cargo test -p discopt-core`. This is a certified-path soundness
fix — run the differential-bound checks per CLAUDE.md §5.

**Phase 2 — confirm MR-1.** Once TG-1 is fixed, re-audit `_fbbt_argument_box`; add
a heterogeneous-array univariate-rescue model to confirm the envelope is now valid
(and add it as a regression test regardless).

**Note.** TG-1 is a second instance of an array/element-collapse class distinct
from but rhyming with the extraction "array-as-sum" bug (solver-core CORE-1): both
come from code that assumes a variable block is scalar. Worth a codebase sweep for
other `.first()`/`.flat[0]`/`treat-block-as-scalar` sites on array variables.
