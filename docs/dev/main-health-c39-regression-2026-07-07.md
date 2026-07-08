# MAIN-HEALTH: C-39 follow-up — genuine `≥`-infeasible node LPs must certify (restore green main) (2026-07-08)

**Verdict.** Main went RED at #554 (C-39, the dual-simplex false-infeasible fix)
on the "Python fast" job: six `test_amp.py` piecewise/trig-relaxation tests began
asserting `'iteration_limit' == 'optimal'`. The mechanism is **not** cycling in
`drive_out_basic_artificials`, and **not** over-iteration — it is **C-39's
Farkas-gated `Infeasible` failing to certify a genuinely-infeasible node LP**,
because the `slack_upper_bounds` recovery C-39 added handled only `+1` (`≤`)
slacks, not `−1` (`≥` surplus) slacks. The AMP piecewise/trig-square relaxations
emit `≥`-constraints (secant / super-level bounds) → `−1` surplus slacks; when the
phase-1 Farkas ray touches one of their open upper sides, the Neumaier–Shcherbina
`g₀` collapses to `−∞`, the certification bails, and the engine returns the honest
non-fathoming `Numerical` instead of the verified `Infeasible`. The MILP B&B then
cannot prune the infeasible node, grinds to the node cap, and the solve reports
`iteration_limit`.

Fix: extend `slack_upper_bounds` to also recover the exact, superset-preserving
upper bound for a `−1` surplus slack (`s_i = (A x)_i − b_i ≤ max_x (A x)_i − b_i`),
symmetric to the existing `+1` case (`s_i = b_i − (A x)_i ≤ b_i − min_x (A x)_i`).
Rust-only (`crates/discopt-core/src/lp/simplex/primal.rs`); no Python touched.
The C-39 false-infeasible fix is preserved and the two coexist: `Infeasible` is
still returned **only** with a verified Farkas ray; the slack bound is exact and
one-sided, so it can never make a *feasible* LP certify.

## Part 1 — reproduction + root-cause

Branch `fix-main-health-c39-regression` from `origin/main` (`056fe087`). Build:
`maturin develop --release` (`_rust.cpython-312-darwin.so` 2.85 MB). Env
`PYTHONPATH=<wt>/python JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`. Corpus
`~/Dropbox/projects/discopt-minlp-benchmark/`, oracle `minlplib.solu`.

**Reproduced** (`pytest python/tests/test_amp.py::…`): `AssertionError: assert
'iteration_limit' == 'optimal'` on the six named tests. Confirmed C-39 is the
cause (passed at #553, fails at #554).

**Instrumented the engine (env-gated `eprintln!`, all removed before commit):**
- The regressed solve is the MILP piecewise relaxation, which has integrality, so
  the warm-LP fast path is skipped and it runs the **MILP B&B**; the node LP
  relaxations are what I instrumented.
- `drive_out_basic_artificials` does **not** cycle: 496 calls, every one
  early-returns with `pivots ≈ 0` (marks a row `stuck`, no reducing structural
  column). No primal `IterLimit` fires. The residual sometimes even *rises* under
  a degenerate step — but this path is not the failure route.
- After a fresh refactorization at the phase-1 stop, **0 improving columns**
  (`best |dⱼ| = 0`): phase-1 is at a genuine positive optimum of `min Σ artificials`
  → the LP is genuinely infeasible (or the ray must certify it).
- **HiGHS ground truth** on the captured node LP (`m=159, n=186`, dumped via a
  temporary env-gated hook): `Infeasible`, and HiGHS supplies a dual ray with
  `g₀(+y) = 0.0386 > 0` and **0 open-box columns** — a valid, certifiable Farkas
  ray.
- **Our** phase-1 ray gives the *same* `g₀ = 0.0386 > 0` but reports **`n_open = 1`**:
  the certification hits an open column and bails the sign. The open columns are
  the last 12 (cols 174–185), each a single-entry column with coefficient **`−1`**
  on rows 147–158 — `≥`-constraint **surplus** slacks. `slack_upper_bounds`
  recognized only `+1` slacks, so these stayed `+∞` → `open → g₀ = −∞` → bail →
  `Numerical` → lost fathom → node-limit → `iteration_limit`.

All three failing-test LPs I captured are HiGHS-**infeasible**. Class: **lost
fathom** (a genuine infeasibility the certificate check cannot verify), the mirror
of C-39's false-infeasible.

## Part 2 — the fix

`slack_upper_bounds` now accumulates both `min_x (A x)_i` and `max_x (A x)_i` over
the structural columns and recovers:
- `+1` slack (`Σ A_ij x_j + s_i = b_i`): `s_i ≤ b_i − min_x (A x)_i` (unchanged).
- `−1` surplus slack (`Σ A_ij x_j − s_i = b_i`): `s_i ≤ max_x (A x)_i − b_i` (new).

Each is exact (one pass, no FBBT division roundoff) and superset-preserving, so it
removes only already-infeasible points and can never make a *feasible* LP certify.
Any slack whose row coefficient is not exactly `±1`, or whose relevant structural
side is genuinely open, keeps `+∞` (the check conservatively bails that sign).

## Part 3 — verification (every gate)

- **`test_amp` 6/6:** `test_continuous_trig_square_uses_direct_piecewise_relaxation[sin-sin]`
  / `[cos-cos]`, `test_mixed_curvature_affine_trig_uses_piecewise_relaxation[sum--4.0]`
  / `[neg_sum--4.0]`, `test_gas_square_difference_tightening_strengthens_root_relaxation`,
  `test_partitioned_square_secants_tighten_circle_superlevel_bound` — all `optimal`.
- **Full `pytest python/tests/test_amp.py`:** 135 passed, 0 failed.
- **`cargo test -p discopt-core`:** 427 passed (was 426; +1 new regression test).
  New: `c39_surplus_slack_infeasible_certifies` — vendored fixture
  `testdata/c39_surplus_slack_infeasible_lp.json` (the captured `m=159, n=186`
  HiGHS-infeasible node LP with 12 `−1` surplus slacks) asserts the cold solve
  returns `Infeasible`; **FAILS pre-fix** (returns `Numerical`, verified by
  neutralizing the `−1` branch), passes after.
- **C-39 HiGHS-agreement battery still clean:** re-solved ex1224 / kall_circles
  c6a/c7a/c8a / nvs01/02/06/08 (25 s / 8000-node caps) and checked vs the
  `minlplib.solu` oracle: **0 false-infeasible, 0 dual-bound-crosses-oracle**.
  ex1224 / nvs01 / nvs02 / nvs06 / nvs08 certify to the exact oracle optimum; the
  kall_circles instances run out the (small) budget as `feasible`/`time_limit` —
  crucially never false-infeasible or bound-crossing. The `−1` recovery reintroduces
  no false-infeasible.
- **`check_cert_neutrality.py`:** **NEUTRAL** (exit 0), all 41 rows pass. Notably,
  the fix *restores* cert-neutrality on **nvs02** and **nvs12**, which were
  **regressed on bare `origin/main`** by the very same lost-fathom mechanism:
  - Bare main (fix stashed): 4 violations — nvs02 `nodes 101→799`, `|Δobj|=2.02e-07`
    (objective drift); **nvs12 `status=feasible` (lost certificate)**; nvs13
    `nodes 19→49`.
  - With the fix: nvs02 `101→101 |Δobj|=0`, nvs12 `195→195 optimal`, nvs13 `49→49`.
  This is direct evidence the fix addresses the *class*, not just the AMP instance.
- **cert-baseline per-row disposition (as instructed, verified each — no blanket
  regen):**
  - **nvs02 (101):** benign — the fix restores it to the committed baseline exactly
    (`101→101, |Δobj|=0`). No rebaseline; the flag was the bare-main regression the
    fix removes.
  - **nvs12 (195):** benign — restored to baseline exactly (`195→195, optimal,
    |Δobj|=0`), recovering the optimality certificate bare main had lost. No
    rebaseline.
  - **nvs13 (19):** benign, **objective bit-identical** (`-585.2`, `|Δobj|=0`),
    deterministic `node_count = 49` (3/3 runs). This node drift is **pre-existing on
    main** (present with the fix stashed too), documented as main-drift #550/#551/#552
    — not introduced here. Regenerated this single row (`node_count 19 → 49`,
    objective unchanged) with this justification.
- **`pytest -m slow test_adversarial_recent_fixes.py`:** 10 passed.
- **`pytest -m smoke` (full python/tests):** 626 passed, 9 skipped, 0 failed.
- **`cargo clippy -p discopt-core`:** clean. **`cargo fmt --check`:** clean.
- **`ruff check` / `ruff format --check`:** clean (no Python touched).
- **pre-commit `mypy` hook (pinned v2.1.0):** Passed.

## CI apt/Ipopt flake (secondary)

Noted, not scoped into this PR. The "Python fast" job intermittently fails
installing Ipopt via `apt` (packages.microsoft.com fetch). This PR restores the
*test* green; the transient apt outage is an orthogonal CI-hardening item (add a
retry to the apt step, or `pytest.importorskip` the Ipopt/GAMS-dependent tests so a
dep outage skips rather than reds). Recommended as a separate follow-up to avoid
scope-creep on this restore-green-main change.

## Files

- `crates/discopt-core/src/lp/simplex/primal.rs` — `slack_upper_bounds` extended
  for `−1` surplus slacks (`max_ax` accumulation + `coef ≈ −1` branch); doc updates
  on `slack_upper_bounds` / `farkas_ray_certifies_cols`; new regression test
  `c39_surplus_slack_infeasible_certifies`.
- `crates/discopt-core/src/lp/simplex/testdata/c39_surplus_slack_infeasible_lp.json`
  — captured HiGHS-infeasible node LP fixture (35 KB, 12 `−1` surplus slacks).
- `docs/dev/data/cert-baseline.jsonl` — nvs13 `node_count 19 → 49` (objective
  identical; pre-existing main-drift #550/#551/#552).
