# Infra / Interop Review — solver_tuning, callbacks, result_io, profiles, constants, pyomo, cutest

**Date:** 2026-07-03
**Scope:** `solver_tuning.py` (169), `callbacks.py` (213), `result_io.py` (164),
`profiles.py` (103), `constants.py` (50), `pyomo/` (`_mapping`, `_writer`,
`solver` — 467), `interfaces/cutest.py` (531). Also surfaced a concrete
`solver.py` lazy-constraint bug while tracing the callback path.
**Method:** Delegated verification; testable paths exercised empirically. The two
correctness findings (INT-2, INT-1) re-confirmed here. Existing tests pass — none
cover the gaps found.

Headline: one **new P1 correctness bug** on the `nlp_bb=True` + `lazy_constraints`
path (cuts silently dropped *and* the excluded point accepted), plus a re-confirmed
`from_nl` binary-bounds bug now shown to hit the Pyomo bridge and the MINLPLib
corpus. The config/serialization plumbing is otherwise sound (certificate flags
round-trip correctly).

---

## 1. Summary

| # | Severity | Component | Finding |
|---|----------|-----------|---------|
| INT-1 | **P1 correctness** | `solver.py:7479` + `1461-1478` | On the **NLP-BB path**, `_invoke_pre_import_callbacks` is called with `_cut_pool=None` while forwarding a live `lazy_constraints` callback. `_cut_pool.add(...)` raises `AttributeError`, swallowed by the broad `except`; because `.add` precedes the node-rejection line inside the `try`, the rejection is **also lost** — the integer-feasible point the cut meant to exclude is **accepted as incumbent** [CONFIRMED: code path + agent end-to-end repro returns `y=0` instead of `y=1`] |
| INT-2 | **P1 correctness** | `modeling/core.py:3629-3630` (`from_nl`) | Binary columns drop the `.nl`'s `lb/ub` (the integer branch keeps them). A `.nl` binary with `lb=ub=1` re-imports as free `[0,1]` → wrong optimum. **Hits the Pyomo `SolverFactory('discopt')` bridge and any `from_nl` of a MINLPLib instance with fixed binaries** [CONFIRMED; = modeling-review M2, here with corpus/bridge impact] |
| INT-3 | P2 | `pyomo/solver.py:164-171` | Pyomo bridge stores the **incumbent** in `problem.lower_bound` and the **dual bound** in `upper_bound` — swapped for minimize (incumbent is an *upper* bound). Only observable with a nonzero gap; misreports `SolverResults` bounds but does not corrupt the loaded primal [CONFIRMED by logic] |
| INT-4 | P2 | `callbacks.py` + `solver.py:1470-1477` | The same swallow-before-reject ordering means **any** exception in `cut_result_to_dense` (e.g. a cut referencing an unknown variable) drops the cut *and* accepts the node — a malformed user cut yields silent acceptance rather than a loud error [CONFIRMED by inspection] |
| INT-5 | P3 | `profiles.py:98` | `resolve_options` does `cli_overrides.pop("tuning")` — **mutates the caller's dict**; a caller looping over profiles loses `tuning` after the first call [CONFIRMED] |
| INT-6 | P3 | `result_io.py:26-49` | Serialization omits `root_bound/root_gap/root_time/solver_stats` → `None` after save/load. **Certificate fields round-trip exactly** (`status/objective/bound/gap/gap_certified/nlp_bb/subnlp_*`) and `__post_init__`'s soundness downgrade re-applies on load — only displayed instrumentation is lost [CONFIRMED] |
| INT-7 | P3 suspected | `interfaces/cutest.py:282-304` | `evaluate_lagrangian_hessian` assumes `pycutest.hess(x,v)` = `∇²f + Σvᵢ∇²cᵢ` (plus sign); if CUTEst uses `L = f − y·c` the multiplier sign flips. pycutest not installed — unverifiable here; the obj_factor split is algebraically correct given the assumed convention [SUSPECTED] |

Verified **correct**:

- **`solver_tuning.py`** — field validation (bounds on the RLT/stride ints,
  `node_bound_mode ∈ {lp,milp}`), `replace()` unknown-field rejection, frozen-ness,
  per-construction env resolution, ContextVar publish/reset all behave correctly.
  (Note: a malformed `DISCOPT_*` env var now hard-fails every `SolverTuning()`
  construction — fail-loud, arguably intended.)
- **`constants.py`** — `INFEASIBILITY_SENTINEL(1e30) > SENTINEL_THRESHOLD(1e29) >
  CONSTRAINT_INF(1e20)` ordering consistent; no wrong constant.
- **`callbacks.py`** — `CutResult` sense validation and `cut_result_to_dense`
  index math correct; user cuts not validated for soundness before acceptance,
  which matches commercial lazy-constraint contracts (user's responsibility) — the
  real defect is the swallow-and-accept (INT-1/INT-4), not the absence of
  validation.
- **`pyomo` bridge** otherwise: sense handling (no spurious sign flip on maximize),
  status-enum mapping (all statuses resolve), and `m.y.fix(1)` (Pyomo substitutes
  the value, the var isn't emitted as a column — so the common fix path is safe;
  INT-2 needs a binary emitted *as a column* with narrowed bounds).
- **`result_io`** certificate soundness — the `__post_init__` downgrade
  (non-finite bound ⇒ `gap_certified=False, bound=None`) re-applies idempotently on
  load; a legitimately-uncertified result stays uncertified.

---

## 2. The two correctness bugs

**INT-1.** Confirmed the code: `solver.py:7479` passes `_cut_pool=None` into
`_invoke_pre_import_callbacks` on the NLP-BB route, while the same call forwards the
user's `lazy_constraints`. In the callback body (`1461-1478`) the sequence is
`_cut_pool.add(...)` → `result_lbs[i] = _INFEASIBILITY_SENTINEL` → `continue`, all
inside one `try`; the `.add` on `None` raises before the sentinel assignment, and
the broad `except` (`1477`) downgrades it to a warning. Net: with `nlp_bb=True`,
lazy constraints are **silently ignored and the point they exclude is accepted as
the incumbent** — a wrong answer. The docstring (`3747`) even says lazy constraints
need the spatial-B&B cut pool, but the explicit `nlp_bb=True` route (`3481`) has no
guard. **Fix:** either build a real cut pool on the NLP-BB path, or **refuse
loudly** when `lazy_constraints` is combined with `nlp_bb=True` (per the
"refuse rather than silently approximate" rule); and move the node-rejection
*before* the `.add` so a cut-construction error can never yield acceptance (INT-4).

**INT-2** is modeling-review M2, but the infra pass adds the impact surface: it is
inherited by the Pyomo `SolverFactory('discopt')` bridge and by any direct
`from_nl` of a MINLPLib `.nl` carrying a fixed binary — i.e. it can silently
mis-model corpus instances the correctness gate is meant to police. Fix in
`from_nl`: route a bound-narrowed binary through `m.integer(name, lb, ub)` (or
extend `Model.binary` to accept bounds).

---

## 3. Plan (for Opus)

**Phase 1 — `fix(correctness): INT-1..INT-2` + INT-4.** NLP-BB lazy-constraint
path: build the cut pool or refuse loudly; reorder reject-before-add so no swallowed
exception accepts a node. `from_nl` binary bounds (shared with the export/modeling
fix — one change closes M2 + INT-2). Regression tests: the `nlp_bb=True` +
lazy-cut repro must return `y=1`; the fixed-binary `.nl` round-trip must preserve
the optimum.

**Phase 2 — `fix(interop): INT-3, INT-5, INT-6, INT-7`.** Un-swap the Pyomo
`lower_bound`/`upper_bound` by sense; copy-don't-mutate in `profiles.resolve_options`;
add the instrumentation fields to `result_io._SCALAR_FIELDS`; verify the CUTEst
Hessian sign convention against a pycutest reference (install-gated test).
