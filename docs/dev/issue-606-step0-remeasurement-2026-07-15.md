# Issue #606 Step-0 re-measurement — probe class deleted by the #632 cutover; pathology unmeasurable at main

**Date:** 2026-07-15 · **Base:** `main` @ `7f190df` (post #632/#636 uniform-engine
cutover, post baron-gap G1–G6 stamps) · **Measurement-only** — no solver changes.
Runs in a fresh Linux container (4 cores, release-mode Rust ext); wall times
indicative, node/pivot/status counts deterministic.

## 0. TL;DR

Issue #606 (split from #598) tracks the in-house MILP B&B's node-efficiency gap,
measured 2026-07-10 by DECOMP-1 on the AMP multi4N partitioned-lift MILP:
**459 nodes / 22,833 pivots / 81 % degenerate dual pivots vs 19 HiGHS nodes**.

**Verdict: the probe no longer reproduces — not because the gap was fixed, but
because the #632/#636 cutover deleted the instance class.** The canonical
construction (`_make_alpine_multi4n(n=2, exprmode=1)` → `build_milp_relaxation`,
`n_init=2`) now emits a **pure LP** (23 cols / 70 rows / **0 integer columns**;
was 161 / 582 / **44**): `build_milp_relaxation` delegates to the uniform
factorable engine and **ignores `disc_state`**, so no partition binaries are
built. The in-house engine solves it in **one node LP, 0.023 s, 0 B&B** — there
is no MILP B&B left to be inefficient on this path.

Two consequences discovered while re-measuring (both bigger news than #606):

1. **The relaxation is 4.4× looser on this instance** — lifted-MILP optimum
   −26.822045 (HiGHS-verified, 2026-07-10) vs LP bound **−118.154** today
   (HiGHS agrees on the identical matrix: sound, just loose).
2. **`amp-integration` CI is red on `main`** — every run since the #636 merge
   (`0f3ebd7`, 2026-07-15T14:04Z; last green `016313c7`, 07-14T23:04Z) fails:
   **22 bulk failures** (multilinear-lift KeyErrors, the multi4n bound gate
   asserting −26.822045, partition/SOS2/embedding tests) **+ the 3
   `amp_cert_heavy` tan/abs certification tests**, which now end
   `'feasible' != 'optimal'` at `gap_tolerance=1e-6`, TL = 300 s (run
   29435960604 @ `8567dae`; reproduced locally at `7f190df`). These 25 are *not*
   in #640's parked `_CUTOVER_DEFERRED_TESTS` list. With `disc_state` ignored,
   AMP's partition-refinement loop no longer tightens the relaxation between
   iterations, so certification at tight gaps has no mechanism to converge.

## 1. Old vs new, same probe

| metric (amp_multi4n probe) | 2026-07-10 (`16f2550`) | 2026-07-15 (`7f190df`) |
|---|---:|---:|
| matrix (cols / rows / ints) | 161 / 582 / 44 | 23 / 70 / **0** |
| in-house solve | 459 nodes, 22,833 pivots, `iteration_limit` @ 0.85 s | 1 node LP, ~16 pivots, `optimal` @ 0.023 s |
| degenerate dual pivots | 18,511 (81 %) | 16 (trivial) |
| RefacCap trips | 443 | 0 |
| relaxation optimum (min sense) | −26.822045 (MILP) | **−118.154** (LP) |
| scipy/HiGHS contrast | 19 nodes, 0.24 s | LP: no B&B, ~1–3 ms |

HiGHS feature-attribution toggles (the experiment #606 queued: presolve on/off
via `scipy.optimize.milp`; presolve × symmetry via `highspy` 1.15.1) are **moot**:
with 0 integer columns every configuration solves the LP in milliseconds to the
same −118.154. There is no node gap left to attribute.

## 2. Is the in-house MILP B&B engine still live?

Yes — `get_milp_solver(backend="auto")` remains **simplex-first**
(`python/discopt/solvers/lp_backend.py`), so any matrix relaxation with genuine
integer columns (models with original integer variables) still routes to the
Rust MILP B&B (`crates/discopt-core/src/bnb/milp_driver.rs`). But on today's
default path those MILPs are trivial: profiling the mixed-integer tan/abs cert
tests (`nlp_mi_004_010/011`, one integer variable) shows the entire MILP phase
at ~1 node LP / ~5 degenerate pivots / < 1 ms per solve. The
"pathologically slow in-house MILP B&B (#606)" attribution in
`python/tests/conftest.py` (`amp_cert_heavy` marker) and
`.github/workflows/amp-integration.yml` STEP 2 is **stale**: those tests are now
slow-and-failing because certification never converges (tightness), not because
the MILP B&B grinds (speed).

## 3. Implications for #606

- The **measured defect signature is unmeasurable at main**: the
  partitioned-lift MILP class that produced 81 % degeneracy + 24× nodes exists
  only behind the deleted federation path. Anti-degeneracy (EXPAND/perturbation)
  and branching-quality work scoped to that class has no current reproducer and
  should not be built against a hypothesis (CLAUDE.md §4).
- The gap **returns to scope if/when piecewise partition lifts return** (the
  "product-side tightness parity" polish deferred by #632, adjacent to #640) —
  which the red multi4n bound gate and dead AMP refinement loop suggest is not
  optional polish but required for AMP certification. Recommended disposition:
  re-scope #606 as *conditional on* the partition-lift restoration (or fold it
  into that work's acceptance criteria: restored lifts must not resurrect the
  459-node pathology), rather than leaving it as an open work item pointing at
  a deleted code path.
- Residual simplex-numerics concerns (refactor-cap churn, degeneracy on
  ill-conditioned LPs) are now tracked by **#649** (G6 escalation) with its own
  reproducer (bchoco06) — that is the live home for the Rust simplex hardening
  half of #606's recommendation.

## 4. Falsifications recorded

- "The #606 probe (459 nodes / 81 % degenerate / 24× vs HiGHS) reproduces at
  main" — **falsified 2026-07-15**: the construction emits a 0-integer LP;
  1 node LP, `optimal`, 0.023 s. Cause is class deletion (#632 cutover ignoring
  `disc_state`), not an efficiency fix.
- "The `amp_cert_heavy` tests are slow because the in-house MILP B&B is
  pathologically slow (#606/#608)" — **stale at main**: MILP phase < 1 ms;
  the tests now fail (`'feasible' != 'optimal'`) because the static uniform
  relaxation cannot close a 1e-6 gap without partition refinement.

## 5. Reproduction

```bash
# In-house probe (DECOMP-1 lane, fresh subprocess, Rust profile counters):
DISCOPT_PROFILE=1 python scripts/decomp1_cert_effort_drive.py 60 amp_multi4n

# tan/abs cert tests (the amp_cert_heavy CI step):
DISCOPT_PROFILE=1 pytest python/tests/test_amp_integration.py -v --timeout=900 \
  -p no:cacheprovider -m amp_cert_heavy -k tan_abs

# CI evidence: workflow "AMP integration", run 29435960604 (main @ 8567dae),
# bulk step: 22 failed / 159 passed; heavy step: 3 failed.
```
