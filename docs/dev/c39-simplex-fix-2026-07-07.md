# C-39 fix — the in-house simplex must not return false-infeasible on a feasible LP (2026-07-07)

**Verdict:** the recurring numerical false-infeasible (root cause behind C-38's
false-optimal and C-42's cut-inherit truncation) is a **missing phase-1 completion
step** in the in-house two-phase primal simplex. On the heavily-redundant lifted
McCormick relaxations, phase-1's reduced-cost optimality test terminates with an
**artificial still basic at a positive value** while every structural column that
could expel it is priced at reduced cost ≈ 0 (already spanned by the redundant
rows) — so the strict `dⱼ < −tol` entering rule never selects it, `Σ artificials`
stalls above zero, and the LP is falsely declared *infeasible* even though it is
feasible (HiGHS solves the identical LP). The engine returned `LpStatus::Infeasible`
on the sole test `Σ artificials > 1e-6`, with **no verified Farkas ray**. Fix:
(1) complete phase-1 by driving every removable basic artificial out with
feasibility-preserving ratio-test pivots, and (2) gate *every* `Infeasible` return
on a **verified Farkas certificate** (Neumaier–Shcherbina `g₀(y) > 0`, with exact
slack-upper-bound recovery for the open slack columns), on both the cold primal and
warm dual paths. Invariant
established: **`status = Infeasible` is returned ONLY with a verified Farkas ray;
otherwise the engine returns a certified optimal/feasible or an honest `Numerical` —
never a false infeasible.**

Branch `fix-c39-simplex-false-infeasible` from `origin/main` (`4e62b56e`).
Rust-only change (`crates/discopt-core/src/lp/simplex/{primal,dual}.rs`); no Python
touched. Build: release (`maturin develop --release`, pounce `_pounce.abi3.so`
4.73 MB). Env `PYTHONPATH=<wt>/python JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`, corpus
`~/Dropbox/projects/discopt-minlp-benchmark/`, oracle `minlplib.solu`.

## Part 1 — captured LPs + root-cause (entry experiment)

**Captured (kall_circles_c8a default path):** dumping every uncertified `infeasible`
verdict of the node-LP boundary (`solve_lp_warm_std`) during the solve captured **9**
lifted McCormick node LPs. Fed each to the in-house simplex (cold + warm) and to
HiGHS:

| LP | coef spread | Rust cold | Rust warm | HiGHS |
|---|---:|---|---|---|
| kall_c8a_0..8 | 4.6e2 – 9.8e3 | `infeasible`, farkas=**False** | `infeasible`, farkas=**False** | **feasible (Optimal)** |

All 9 were FALSE (HiGHS feasible; every bound satisfied, max row residual 3.7e-14).
Conditioning is only ~1e2–1e4 — below the Python equilibration re-verify trigger
(1e3), and even the *cold* solve fails, so the fault is not warm-start staleness or
scaling.

**Root-cause step (`crates/discopt-core/src/lp/simplex/primal.rs`):**

- `simplex_loop` (phase 1) declares optimality at `primal.rs:1217`
  (`None => return Ok(())`) when no column has reduced cost `< −tol`. Instrumenting
  the stop on `kall_c8a_2` (m=4456): phase-1 ran **2956 pivots**, then stopped with
  **worst reduced-cost improvement 1.5e-16** (genuinely no improving column) yet a
  fresh `basic_values` gave **`Σ artificials = 378` across 16 basic artificials** —
  with a **healthy LU** (growth 1.0). A fresh cold refactorization did not change the
  reduced costs, ruling out LU staleness.
- Probing each of the 16 basic artificials for a nonbasic *structural* column with a
  nonzero pivot in its row: **11 were expellable, 0 truly stuck.** A degenerate
  (zero-reduced-cost) pivot would drive each out and reveal feasibility — HiGHS does
  exactly this cleanup; the in-house phase-1 did not.
- The old infeasibility test (`if infeas > 1e-6 { return Infeasible }`, pre-fix) then
  declared `Infeasible` on this feasible LP, with a Farkas ray candidate that
  **fails to verify** (`g₀(±y) = −163, −27 < 0`).

Class: **(a) unsound per-node relaxation** — an `Infeasible` verdict trusted without
a certified Farkas ray, exactly the C-38 finding, now traced to the simplex source.

## Part 2 — the fix (Rust, general — the class, not the instance)

**(1) Phase-1 completion — `drive_out_basic_artificials` (`primal.rs:795`).** After
`simplex_loop`, if `Σ artificials > 1e-6`, drive every basic artificial out of the
basis with **bounded-variable ratio-test pivots** (Harris two-pass, largest-pivot
leaving choice, Bland tie-break, degenerate-run + stuck-row guards for termination).
Every pivot is a standard step *inside the phase-1 feasible region* (artificials in
`[0,∞)`, structurals at a bound), so it preserves `A x = b` and phase-1 feasibility
and never raises the artificial sum: a feasible LP is never turned infeasible nor the
reverse — it only completes the phase-1 the reduced-cost test left short. Artificials
cleared ⇒ the LP is feasible and the basis is primal-feasible for phase 2 (a real
bound is produced); a residual that survives ⇒ a genuine redundant-row infeasibility.

**(2) Farkas-gated `Infeasible` (`primal.rs:703–747`, `dual.rs:526`).** An
`Infeasible` is returned **only** when a phase-1 dual ray `y = B⁻ᵀc₁` *verifies*
emptiness via the Neumaier–Shcherbina `c = 0` safe bound `g₀(y) > 0` — checked
before the cleanup (genuine infeasibility carries the ray on the phase-1-optimal
basis, so no fathom is lost) and re-checked after. On the warm **dual** path the same
verification gates the `Infeasible` return (`dual.rs`); an unverifiable ray returns
`None`, routing to the trusted cold two-phase solve. Anything unproven is the honest,
non-fathoming `Numerical`.

**(2a) Slack open-bound recovery (`farkas_ray_certifies_cols` / `slack_upper_bounds`,
`primal.rs`).** A `≤`-constraint LP's Farkas ray has nonneg row multipliers whose
reduced cost on a slack `s_i ∈ [0, ∞)` selects its **open** upper side, collapsing
`g₀` to `−∞` — so 100 % of the genuinely-infeasible node LPs (whose structural
columns are all finitely bounded on these relaxations, leaving slacks the only open
columns) would fail to certify. Each slack sits in exactly one row `i` with
coefficient `+1`, so `s_i = b_i − Σ_j A_ij x_j ≤ b_i − min_x Σ_j A_ij x_j` — a finite
bound from the structural box, computed in **one exact pass** (no iterative-FBBT
division roundoff). This is superset-preserving (removes only already-infeasible
points), so it can never make a *feasible* LP certify: sound by construction.
**This step is load-bearing:** without it the engine returned `Numerical` on
genuinely-infeasible nodes (nvs01/ex1224/nvs08: 12/17/20 lost fathoms each), losing
the certificate on the cert panel; with it every genuine fathom is recovered and the
panel is neutral. (An earlier iterative-FBBT port was *rejected*: its accumulated
division roundoff produced false certificates that HiGHS/the Python check reject —
the exact-slack-bound version has no such roundoff and is provably one-sided.)

The C-38/C-42 Python-side guards are unchanged (defense-in-depth).

## Verification

**Captured LPs now correct.** Every one of the 9 kall_circles false-infeasible LPs
returns non-`Infeasible` (feasible/Optimal or honest `Numerical`) on both cold and
warm paths; HiGHS agrees each is feasible.

**HiGHS-agreement battery** (all node LPs across kall_circles_c6/c7/c8,
kall_congruentcircles, nvs06/17/19/23/24, cross-checked vs HiGHS): **0
false-infeasible, 0 false-feasible** over ~1600 `optimal`/`infeasible` verdicts
(pre-fix: ≥7 false-infeasible on the same battery). Genuine-infeasible fathoms:
**0 lost** (nvs01/ex1224/nvs08, previously 12/17/20 lost, recovered by the FBBT step).

**Recovered behaviour.** kall_circles_c8a default path: no false-optimal (dual bound
`−1e-9 ≤ 2.5409` oracle). The regressed-then-recovered cert-panel instances all
certify at the baseline again:

| instance | fix status | obj | oracle | nodes (baseline) |
|---|---|---:|---:|---:|
| ex1224 | optimal | −0.9434705 | −0.9434705 | 53 (53) |
| nvs01 | optimal | 12.4696688 | 12.4696688 | 17 (17) |
| nvs02 | optimal | 5.96418452307 | 5.9641845 | 101 (101) |
| nvs08 | optimal | 23.4497274 | 23.4497274 | 57 (57) |
| nvs12 | optimal | −481.2 | −481.2 | 195 (195) |
| st_e29 | optimal | −0.9434705 | −0.9434705 | 53 (53) |

**nvs06 `DISCOPT_CUT_INHERIT=1`:** the cut-augmented root node LP no longer returns a
*false* `Infeasible` (the C-39 root cause is gone); it returns the honest `Numerical`
(feasible-but-no-clean-vertex on that degenerate augmented system). Certifying it
*without a bound* still needs the driver-side C-42 pool-strip (a separate,
driver-layer fix not on this base); that workaround remains the belt-and-suspenders
the task specifies. C-39 removes the false-infeasible source; C-42 handles the driver
sentinel-on-skip. Not a regression — flag-ON nvs06 is unchanged from the base.

**No regression.**
- `cargo test -p discopt-core`: **426 passed** (incl. the pre-existing
  `warm_incremental_matches_cold_under_many_pivots` genuine-infeasible test, which now
  agrees cold==warm again, and `infeasible_detected` / `batch_unscales_ray_like_single_solve`
  genuine-infeasible tests).
- New Rust regression tests (`primal.rs` test module):
  `c39_kall_circles_captured_lp_not_false_infeasible` (vendored 447 KB fixture
  `testdata/c39_kall_false_infeasible_lp.json`, HiGHS-feasible) asserts the cold solve
  is never `Infeasible` — **FAILS on the pre-fix engine** (returns `Infeasible`),
  passes after; plus `c39_redundant_row_not_false_infeasible` (a tiny synthetic
  redundant-row LP).
- `check_cert_neutrality.py` (41-panel): **byte-identical except the single
  pre-existing `nvs13` node_count 19→49 drift** already recorded as main-drift
  (#550/#551/#552) — verified to be identical on the bare branch point (this diff
  stashed → same 1 violation, same nvs13 disposition). No new violations; all 40 other
  rows `|Δobj|=0.00e+00`, `nodes n→n`. No rebaseline needed.
- `pytest -m smoke` (python/tests): passed.
- `pytest -m slow test_adversarial_recent_fixes.py`: **10 passed**.
- `cargo clippy -p discopt-core`: clean.

## Files

- `crates/discopt-core/src/lp/simplex/primal.rs` — `drive_out_basic_artificials`
  (phase-1 completion), `slack_upper_bounds` + `farkas_ray_certifies_cols` (shared
  Farkas check with exact slack open-bound recovery), `phase1_farkas_ray`,
  Farkas-gated `Infeasible` in `run`; 2 regression tests + `parse_lp_fixture`.
- `crates/discopt-core/src/lp/simplex/dual.rs` — warm-dual `Infeasible` gated on the
  shared Farkas verification (unverified ray → `None` → cold fallback).
- `crates/discopt-core/src/lp/simplex/testdata/c39_kall_false_infeasible_lp.json` —
  captured HiGHS-feasible node LP fixture for the regression test.
