# ENGINE-1 — the RefacFtFail is a zero-pivot breakdown, not an over-conservative guard (2026-07-08, #557)

**Task.** THRU-5 (`thru5-base-node-solve-2026-07-07.md`) root-caused the ~13× BARON
wall gap on the dense-QP class to the in-house dual simplex refactorizing its LU
basis on nearly every pivot, and attributed 89–100 % of those refactorizations to
`RefacFtFail` — the `feral` product-form (Forrest–Tomlin) LU **update returning
`Err`** on the wide dense-column lifted-McCormick basis. ENGINE-1 is the first
concrete swing at that: root-cause the *exact* FT-fail condition, decide whether
the refactor is **accuracy-necessary** or the bail is **over-conservative**, and
either ship a byte-identical improvement or characterize an honest KILL with the
next option.

**Verdict: KILL (honest), with a byte-identical instrumentation win that
*redirects the roadmap*.** The FT-fail is `feral`'s `RefactorCause::TinyPivot`
with pivot magnitude **~0.0 (exactly zero), ~98–100 % of the time** — a
zero-pivot bump breakdown, NOT the tunable element-growth guard (`Growth` count is
**0**). A zero pivot means the incremental single-row FT elimination has no valid
pivot; the from-scratch refactor (full partial pivoting) is **accuracy-necessary**,
so retuning the FT-update stability/fill threshold (roadmap #1a) is a **hard
soundness NO-GO** — there is no threshold that makes dividing by a zero pivot
sound. The OBBT factorization-reuse alternative (roadmap #3 / Phase 3b) is **also
capped**: the storm is measured to live in the **cold-primal node LP, not the OBBT
probe fan**, and the FT-fail **recurs many times *within a single solve*** (a
per-pivot breakdown), so persisting one factorization across probes cannot
amortize it. What ships is the **cause-split instrumentation** that turns THRU-5's
ambiguous "89–100 % RefacFtFail" into the actionable "~100 % zero-pivot
TinyPivot", which kills the highest-ranked roadmap lever and re-points the effort
at the two options that remain (§4).

Build: release (`maturin develop --release`), pounce `_pounce.abi3.so` = 4.73 MB.
Env: `PYTHONPATH=<wt>/python JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`, dense-QP regime
`DISCOPT_SQUARE_SEPARATE=0` (matching THRU-5, to isolate the base LP solve).
`feral = "0.12.0"` (crates.io registry crate — **not** vendored, no `[patch]`), so
its FT-update kernel is not directly editable from this repo.

## Phase 1 — where `feral` is, and the EXACT RefacFtFail condition

`feral` is an external registry dependency (`crates/discopt-core/Cargo.toml:28`,
`feral = "0.12.0"`), source at
`~/.cargo/registry/src/index.crates.io-*/feral-0.12.0`. Its Forrest–Tomlin
column-replacement update is `src/lu/sparse_update.rs`
(`SparseLu::update` / `update_sparse` / `eliminate_pivot_row`). The update returns
`FeralError::NeedsRefactor` on four causes, recorded on `last_refactor()` as a
`RefactorCause` (`src/lu/mod.rs:57`):

| cause | `sparse_update.rs` line | trigger |
|---|---|---|
| `UpdateBudget` | 94–97 | `updates_since_refactor + 1 > max_updates` (feral default 64) |
| `Singular` | 121–130 | spike has no entry at/below its own rank ⇒ rank-order dependent |
| `Growth` | 180–183 | element-growth high-water `‖U‖∞/‖U₀‖∞ > max_growth` (default **1e8**) |
| `TinyPivot` | 454 / 460 / 498–501 | a bump/final diagonal pivot `≤ zero_pivot_tol·‖U₀‖∞` (default `1e-13·‖U₀‖∞`), missing, or non-finite |

discopt's `Ctr::RefacFtFail` (`lp/simplex/primal.rs:1440,1474`) counts *any*
`FeralLU::update(...).is_err()` on the cold-primal pivot path but did **not**
distinguish the four `feral` causes — that ambiguity is exactly why THRU-5 could
not rule between "necessary" and "over-conservative". Note discopt caps updates at
**48** (`updates >= 48`, below feral's 64), so `UpdateBudget` never surfaces from
feral: discopt's own 48-cap trips first as `RefacCap`. Therefore every
`RefacFtFail` is one of `Singular` / `Growth` / `TinyPivot`.

**Entry experiment (the measurement THRU-5 lacked): split `RefacFtFail` by
`feral`'s `RefactorCause`.** Added three `DISCOPT_PROFILE`-gated counters
(`RefacFtGrowth` / `RefacFtTinyPivot` / `RefacFtSingular`) read off
`FeralLU::last_refactor()` at the same trigger site (no-op without the env var; a
cheap field read). Result, dense-QP regime, TL 30 s:

| instance | Refactorizations | RefacFtFail | **Growth** | **TinyPivot** | **Singular** |
|---|---:|---:|---:|---:|---:|
| st_e36 | 139 | 139 | **0** | **139 (100 %)** | 0 |
| nvs24  | 330 | 330 | **0** (1 in a longer run) | **~330 (≈100 %)** | 0 |
| nvs19  | (mostly RefacCap this run) | — | 0 | — | 0 |

**The FT-fail is `TinyPivot`, essentially never `Growth`.** A magnitude histogram
(temporary `DISCOPT_FT_MAG` logging, since removed) pins it down further:

| instance | pivot `mag == 0.0` (exactly zero) | pivot `1e-15…1e-21` (below eps·scale) |
|---|---:|---:|
| st_e36 | 16663 | ~240 |
| nvs24  | 6350 | 115 |

So **~98–100 % of FT-fails are an exactly-zero bump pivot**; the small remainder
sit at or below machine epsilon relative to `‖U₀‖∞`. This is `sparse_update.rs`'s
zero-pivot path (line 454 `mag=0.0`, line 460/498 `|pivot|≈0`), not the
`max_growth=1e8` guard.

## Phase 2 — accuracy-necessary vs over-conservative: NECESSARY

The decision fork is settled by the magnitudes, without needing a residual probe:
a **zero** bump/final pivot is not a cautious-threshold artifact, it is a
numerically **rank-deficient replacement column in triangular-rank order**. The
Forrest–Tomlin scheme eliminates a single pivotal row against the *fixed* upper
bump; when that row's diagonal comes out `0`, the updated factor would divide by
zero — its residual `‖B x − b‖` is unbounded (∞), categorically worse than any
from-scratch refactor. The refactor re-runs full partial pivoting and *re-selects*
a stable pivot ordering the incremental one-row elimination cannot reach. Hence:

* Raising/retuning the FT guard (roadmap #1a) is **unsound**: `zero_pivot_tol` is
  already `1e-13` and the offending pivots are `0.0`; lowering it divides by zero
  (CLAUDE.md §1 hard reject). There is no accurate updated factor to recover.
* The `Growth` guard — the one plausibly over-conservative lever (a large-but-
  bounded growth can still give an accurate factor) — **never fires here**
  (`RefacFtGrowth = 0`), so there is nothing to loosen.

## Phase 3(b) — OBBT factorization reuse is ALSO capped (measured)

The remaining byte-identical candidate was persisting one factorization across the
OBBT probe fan (`PreparedDual` reuse; `dual.rs:172`). Two measurements kill it:

1. **The storm is not in the OBBT fan.** Disabling per-node OBBT
   (`solver._PER_NODE_OBBT_BUDGET_FRAC = 0.0`) leaves st_e36's counters
   **bit-identical** (`RefacFtFail = 42073` on/off over the full solve) — st_e36
   has no dependent-variable structure so per-node OBBT never ran, yet the storm
   is at full strength. On nvs24 (where OBBT does run) turning it off barely moves
   `RefacFtFail`. The storm lives in the **main cold-primal node LP**.
2. **The FT-fail recurs *within* a single solve.** `RefacFtFail ≈ Refactorizations`
   with `DualWarmSolves ≈ 0` and `DualColdFallbacks ≈ 0`: the thrash is the cold
   primal `run_phase` (primal.rs), and a *single* cold solve refactorizes
   dozens–hundreds of times as `feral`'s update returns `TinyPivot` again and again
   across its own pivots. A shared/persisted factorization amortizes the *initial*
   factorize, not a per-pivot mid-solve breakdown — so reuse cannot touch this.

(Full-solve counters, dense-QP regime: st_e36 RefacFtFail 42073, Refactorizations
42074, Phase2Pivots 12263, Phase1Pivots 33695, DualWarmSolves 87,
DualColdFallbacks 0, 153 nodes, 16.6 s, obj −246.0 optimal — identical with OBBT
off.)

## §4 — ranked next options (what an out-of-PR effort must target)

With #1a (retune) and #3 (OBBT reuse) both eliminated for this cause, the leverage
order is:

1. **Cut the dense-column count in the lifted relaxation** (structural, upstream).
   The zero-pivot bump is produced by the dense McCormick-envelope columns: a
   single dense-column basis swap yields a rank-order-dependent spike the one-row
   FT elimination can't pivot. A sparser / column-localized lift shrinks the bump
   so the update rarely hits a zero pivot. **Bound-changing** (differential-bound
   regime). This is now the highest-leverage lever, ahead of any LU-kernel tweak.
2. **A more capable basis-update kernel than single-row FT** (upstream in `feral`,
   or a swap). Bartels–Golub / Suhl–Suhl with an actual pivot search over the bump
   can find a nonzero pivot where the fixed-row FT elimination finds zero. This is
   a `feral`-internal change (external crate; file it upstream as feral#… — the
   discopt-side `RefacFtTinyPivot` counter is the reproducer signal) — NOT
   editable from this repo, and a version bump is not byte-identical.
3. **A production sparse-LP backend** (HiGHS) for the node LP on the dense-QP
   class, keeping the in-house simplex where it wins. Largest scope; only if 1–2
   stall. HiGHS's update kernel handles these bases (it is BARON's order of
   magnitude).

Roadmap #1a (FT-guard retune) is **struck** from the ranked list: proven unsound
for a zero pivot.

## §5 — what shipped

Pure **instrumentation** (all no-ops when `DISCOPT_PROFILE` is unset — the
production hot path is unchanged; the new work is entirely inside
`if crate::profile::enabled()`):

* `crates/discopt-core/src/profile.rs`: three counters
  `RefacFtGrowth` / `RefacFtTinyPivot` / `RefacFtSingular` splitting the existing
  `RefacFtFail` by `feral`'s `RefactorCause`.
* `crates/discopt-core/src/lp/simplex/primal.rs`: at the cold-primal refactor
  trigger, read `FeralLU::last_refactor()` and attribute the FT-fail to its cause.

This is the load-bearing evidence that kills roadmap #1a and #3 and re-points the
effort at §4 #1/#2/#3. The temporary `DISCOPT_FT_MAG` magnitude logging used for
the Phase-2 histogram was removed (no dead knobs).

## §6 — gates (byte-identical + accuracy + suites)

* **Byte-identical cert (41-panel, `check_cert_neutrality.py`):** **41/41 NEUTRAL,
  every `|Δobj| = 0.00e+00`, every node_count identical** (`X→X`). The change is
  gated instrumentation, so with profiling off it cannot move a pivot, bound, or
  node — confirmed empirically.
* **Dense-QP byte-identical:** st_e36 obj −246.0 / node-count and nvs19/nvs24
  objectives unchanged vs `origin/main` across the on/off OBBT measurements.
* **Accuracy (C-39 soundness battery):** `test_gear4_false_infeasible`,
  `test_hda_false_infeasible`, `test_c1_nlp_fathom_infeasible`,
  `test_cert_finite_bound_soundness`, `test_alan_soundness` — **16 passed, 0 new
  false-infeasibles / 0 false-optima**.
* `cargo test -p discopt-core`: **428 passed, 0 failed** (+4 presolve-determinism,
  +1 doctest); `--no-default-features`: **428 passed**.
* `pytest -m smoke python/tests/`: **633 passed, 12 skipped**.
* `pytest -m slow test_adversarial_recent_fixes.py`: **10 passed**.
* `rustfmt --check` on the two touched files: clean. `cargo clippy -p
  discopt-core`: no new warnings.

## §7 — nodes/s before/after

No speedup is claimed: this PR ships instrumentation and a characterized KILL, not
a per-node improvement. The dense-QP nodes/s (st_e36 ≈ 9 nps, nvs24 ≈ 0.3 nps in
the 30 s cap) is unchanged and remains gated by the zero-pivot refactor storm,
whose fix is one of the §4 out-of-PR options. The measurement that would move it
is now instrumented and its lever is identified.
