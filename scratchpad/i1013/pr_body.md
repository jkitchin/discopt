## Summary

Contributes to #1013 (does not finish it — see "What remains" below).

The warm dual simplex has two anti-degeneracy escapes and **neither is reachable
on a lifted relaxation**. Bland's rule engages after `2·(n+1)` consecutive
degenerate pivots; the F2 stall guard trips at the size-derived pivot cap
`20·(m+n)+500`. Measured over a 100-LP panel of in-repo root relaxations (all 9
vendored QPLIB instances and all 66 vendored MINLPLib `.nl` instances,
`rlt_lineq` off and on), `DualBlandActivations` and `DualStallTrips` are **0 on
every single LP** — including cells where 98.7 % of pivots are degenerate and the
solve exhausts its budget. So a degenerate stall had no exit but the iteration
cap, and on one LP it ended in an `infeasible` verdict that HiGHS, every
perturbed arm of this engine, and an elastic `min t` feasibility LP contradict.

This adds `SimplexOptions::dual_stall_patience` (default 2048, from
`DISCOPT_LP_DUAL_STALL_BAIL`): past that many consecutive degenerate pivots the
warm loop returns `None`, i.e. the caller's cold two-phase primal finishes the
solve. It also ships the instrumentation the issue asks for
(`DualDegenerateRunMax` via a new `profile::record_max`, `DualDegenerateRunArms`,
`DualDegenerateStallBails`, and a per-pivot `DISCOPT_LP_DUAL_TRACE` stream).

**Environment caveat, stated up front.** `QPLIB_0911` — the instance the issue is
written around — is unreachable from the environment this was developed in (no
`~/Dropbox` corpus, `qplib.zib.de` blocked by the network policy, and
`scratchpad/i1008/lps/*.npz` was never committed). Its headline cell was **not**
re-measured and is neither confirmed nor contradicted here. Everything below is
measured on the vendored corpora. **This is the main reason to test locally.**

### To run it locally

```bash
python -u scratchpad/i1013/capture.py                 # rebuild lps/*.npz (gitignored, ~888 KB)
python -u scratchpad/i1013/panel.py out.jsonl 20 2 off:DISCOPT_LP_DUAL_STALL_BAIL=0 bail:
python -u scratchpad/i1013/report_panel.py out.jsonl
python -u scratchpad/i1013/tree_panel.py              # tree-level, oracle-checked
```

`capture.py` selects instances by filter and now takes the corpus location from
the environment, so running the issue's own cells needs no source edit:

```bash
I1013_QDIR=~/Dropbox/projects/discopt-minlp-benchmark/qplib \
I1013_NLDIR=~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl \
I1013_ONLY=QPLIB_0911 python -u scratchpad/i1013/capture.py
```

A non-existent directory and a filter matching nothing are both refused.
`DISCOPT_LP_DUAL_TRACE=1` gives the per-pivot stream;
`DISCOPT_LP_DUAL_STALL_BAIL=0` restores the previous loop
(`1`/`true`/`on` mean "enabled at the default patience", **not** a patience of one
pivot — an earlier harness read it literally and produced a phantom 17x tree
regression).

## Correctness

The action on trip is the one every other difficulty in this loop already takes:
return `None` so the caller cold-solves. The cold two-phase primal self-verifies
its own verdict, so the bail can change only *which* engine finishes a solve,
never the value it returns. No tolerance, guard, bound formula or certificate
test is touched. The one status change in the panel is an `infeasible` → `optimal`
**improvement** on an LP independently shown to be feasible.

- [x] Cannot produce a false certificate
- [x] No validation, fallback, or safety guard was weakened to make a check pass

## Tests run (state the result — numbers, not adjectives)

- [x] `pytest -m smoke` → **1008 passed**, 16 skipped, 2 xpassed (re-run after merging `main`)
- [x] `pytest -m slow python/tests/test_adversarial_recent_fixes.py` → **10 passed**
- [x] `cargo test -p discopt-core` → **604 passed**, 0 failed
- [x] `ruff check python/` clean; `cargo fmt --check` clean; `cargo clippy -p discopt-core` clean.
      `ruff format --check python/` reports 2 files (`test_mo_augmecon2.py`,
      `test_log_square_relaxation.py`) — both are unformatted on `main` already and
      untouched here (this branch changes no `python/` file).

## Regression test

Two new tests in `crates/discopt-core/src/lp/simplex/dual.rs`, on captured
fixtures emitted by `scratchpad/i1013/make_fixture.py`:

- `degenerate_run_detector_sees_what_the_stall_trip_counter_cannot` — replays the
  captured `st_testgr3` relaxation (103×144) and asserts `DualStallTrips == 0`
  while the run-length counters record a degenerate run two orders of magnitude
  past the arming threshold. This is the defect: before this change there was **no
  counter that could report it**, so the assertion cannot be written at all
  against the old code.
- `dual_stall_bail_fires_on_a_captured_stall_and_is_bound_neutral` — replays the
  captured `tspn12` relaxation (1345×1801) through the production entry point,
  and carries its own fail-before/pass-after: with `dual_stall_patience = 0` (the
  pre-#1013 loop) it asserts **0 bails**, and with the patience below the LP's
  620-pivot run it asserts exactly **1 bail** *and* the same objective as the
  warm-converged arm.

- [x] Added a regression test that fails before / passes after

## Bound-changing / performance change?

Bound-changing under CLAUDE.md §5 (it changes which engine solves a stalled node
LP, hence the degenerate vertex a bound may come from). It ships **default-ON**
on the strength of the panel below, with `DISCOPT_LP_DUAL_STALL_BAIL=0` as the
opt-out and the legacy path intact.

**LP graduation panel** — 100 captured relaxation LPs, 2 reps, arms interleaved
within each rep, 20 s per-LP limit; reproduced on the final build.

| gate | result |
|---|---|
| unchanged cells | **98 / 100** identical status *and* identical iteration count |
| status regressions | **0** |
| status improvements | **1** — `QPLIB_3814_rlt1` `infeasible` → `optimal` |
| objective drift (optimal/optimal) | **0.00e+00** (n = 96) |
| wall, LPs ≥ 50 ms (10 of 100) | 0.96x–1.20x; the bail fires on **three** cells, gaining 1.13x (`QPLIB_3814_rlt1`), 1.20x (`tspn10_rlt1`) and 1.01x (`QPLIB_3815_rlt1`, neutral) |
| wall, LPs < 50 ms (90 of 100) | 0.74x–1.47x spread at **identical iteration counts** — sub-millisecond noise, not effect |

**Tree-level differential panel** — every vendored `.nl` with a recorded optimum
(16 instances, 60 s each): objective, bound and node count **bit-identical on all
16**; 96 soundness assertions (no bound past its reference optimum, no incumbent
beyond it, no certification or status regression), **0 issues**. The bail does not
fire in these trees (longest degenerate runs 47–87 pivots).

**Falsified and NOT shipped** (measurements in `docs/dev/performance-plan.md` §17
and `scratchpad/i1013/FINDINGS.md`):

| lever | outcome |
|---|---|
| #1008's dual Harris pass, re-landed **armed only inside the stall** (the issue's ask) | fires on 22/100, wall median **0.995x**, iteration median **0**, regresses `tspn10_rlt1` `optimal` → `iter_limit` |
| Bland's rule at a *reachable* run length | fires on 8/100, median 1.007x, same status regression |
| a conjunctive primal-infeasibility progress test | `max_noprog == max_run` on every LP traced — never discriminated once; removed rather than shipped as an untested branch |

The trace also contradicts the issue's hypothesized mechanism: on the worst cell
the chosen pivot magnitude is **exactly 1.0 at the median** with none below 1e-4
and no zero primal steps, so the degeneracy is *dual* and an `|α_rj|` tie-break
has nothing to discriminate on this corpus.

## Corrections to an earlier revision of this description (CLAUDE.md §11)

Three numbers in the panel section were wrong when first posted and are corrected
above. They were re-derived from this PR's own `scratchpad/i1013/final_panel.jsonl`:

| claim as posted | actual |
|---|---|
| "all **68** vendored MINLPLib `.nl` instances" | **66** (`ls python/tests/data/minlplib_nl/*.nl`) |
| "LPs < 50 ms (**83** of 100)" | **90** of 100 — and 90/10 under every aggregation (median/min/max/mean, either arm); the ratio ranges quoted beside it were always computed on the true 90/10 split, so only the count was wrong |
| "the **two** cells where it fires gain 1.13x and 1.20x" | **three** cells fire; the omitted one, `QPLIB_3815_rlt1`, gains **1.006x** — neutral, not a gain |

The third is the one that mattered: dropping the neutral cell made the
fired-cell evidence read as uniformly positive when one of three showed no
effect. The gate rows are unchanged and re-verified — 98/100 identical status
*and* iteration count, 0 status regressions, 1 improvement, objective drift
exactly 0.00e+00 over n = 96 optimal/optimal cells.

Two instruments that could have reported success without measuring anything were
also fixed (see the second commit): `DISCOPT_LP_DUAL_STALL_BAIL` silently read an
unrecognized *off* value as ON — Python's `str(False)` == `"False"` hit exactly
this — and `lprun.py` recorded counters under `if k in snap`, so a stale `_rust`
produced a full set of clean-looking RES lines containing none of the change.
Both now refuse loudly; both were demonstrated fail-before/pass-after against the
real defect, not asserted.

## Related

The `QPLIB_3814_rlt1` verdict is a **separate** defect, filed as #1017: its
Neumaier–Shcherbina margin is built from result magnitudes rather than
accumulation magnitudes, so `bᵀy = 3.0e-8` against a term magnitude of 600 passes
as a proof of infeasibility. This PR only stops a stalled warm loop from being
the thing that reaches that verdict; it does not repair the margin.

## What remains on #1013 — corrected

The earlier text here said the remaining work was to "re-run the issue's five
perturbation cells on `QPLIB_0911` from the full corpus," and attributed the gap
to corpus access. **Both halves of that were wrong**, and the cells have now been
run (`scratchpad/i1013/cells_0911.py`, `cells_0911.jsonl`).

**1. Corpus access was never the blocker.** `QPLIB_0911` is present locally
(`qplib/qplib/QPLIB_0911.qplib`). Captured at 5150×1325, nnz=45146.

**2. The issue's five levers do not exist on this branch.**
`DISCOPT_LP_REFACTOR_INTERVAL`, `DISCOPT_LU_SYMBOLIC_REUSE`, and any LU
`pivot_threshold` knob return zero hits crate-wide. They were `perf/1008` flags
and that branch is unmerged. The cells cannot be run as the issue writes them,
now or after any amount of corpus work. The three rounding-level knobs this tree
*does* have (`DISCOPT_LU_DENSITY_ROUTE`, `DISCOPT_LP_FACTORIZATION_HARDENING`,
`DISCOPT_LP_ITERATIVE_REFINEMENT`) are bit-identical no-ops on this LP — same
objective to 12 decimals, same 553 degenerate pivots, same max run — so there is
no substitute perturbation arm. That result is **inconclusive, not negative**: I
have no arm that perturbs anything, so it says nothing about whether a
perturbation would stall this LP.

**3. The baseline reproduces #1013 exactly** — 1279 pivots, `optimal`,
obj −76.385275000002, `DualDegeneratePivots = 553` (43%), `DualStallTrips = 0`.

**4. But the bail provably cannot fire on `QPLIB_0911`.**
`DualDegenerateRunMax = 40` against a patience of 2048 — off by 51×, and
impossible in principle, since the entire solve is 1279 pivots. Confirming "it
converts that `iter_limit` into a cold-solved `optimal`" was never achievable on
this instance with this patience. 8/8 cells, both arms: identical status,
identical iteration count, zero bails.

### The patience of 2048 is well-placed — here is the measurement

The PR shipped 2048 without presenting evidence for it. The panel supplies it.
`DualDegenerateRunMax` across all 100 LPs, bail OFF:

| LP | max degenerate run | |
|---|---:|---|
| `QPLIB_3815_rlt1` | 7888 | bail fires |
| `tspn10_rlt1` | 5512 | bail fires |
| `QPLIB_3814_rlt1` | 3352 | bail fires |
| — patience **2048** — | | |
| `tspn12_rlt1` | 1274 | |
| `tspn08_rlt1` | 902 | |
| `QPLIB_3871_rlt{0,1}` | 663 | |

min 0, median **7**, p90 341, max 7888. Distribution:
`{0: 2, 1–99: 78, 100–999: 16, 1000–2047: 1, ≥2048: 3}`.

The panel is bimodal — a bulk that never exceeds 1274 and a tail at 3352+ — and
2048 sits in the empty band between them, 1.61× above the highest non-stalling
run and 1.64× below the lowest stalling one. The 3 LPs that exceed the patience
are exactly the 3 cells where the bail fires; there are no near-misses to tune
against.

### Honest read of the graduation case

The mechanism is sound (0 regressions, 98/100 identical, drift 0.00e+00) and the
threshold is well-separated, but it engages on **3 of 100** panel cells, one of
them neutral (`QPLIB_3815_rlt1`, 1.006×). The real win is the single
`infeasible` → `optimal` conversion on `QPLIB_3814_rlt1`. That is a correctness
result and it justifies the change; "broadly net-positive across the panel" would
not be an accurate description of it.

### Instrument defect found and fixed (CLAUDE.md §6)

The first run of these cells produced 32 clean records with **every counter 0**,
including `DualWarmSolves` — impossible in a script whose only entry point is the
warm dual solve. Cause: the counters are gated on `DISCOPT_PROFILE`, which the
driver did not set; without it they read 0 and emit well-formed records that say
"no degeneracy, no bail, no warm solve" about a solve that just did 553
degenerate pivots. My own §8 guard passed the whole way because it checked the
counter symbols *existed*, not that they *fired*. Fixed: `lprun.py` now refuses
on `DualWarmSolves < 1`, and the driver sets `DISCOPT_PROFILE=1`. **The panel in
this PR is unaffected** — `panel.py:49` already sets it; the omission was in my
new driver only.

### Status

`Contributes to` #1013 remains correct, but the reason has changed: not "a cell
is still unrun," but "the cell the issue asks for cannot be constructed on this
branch, and on the instance it names the guard is unreachable by 51×." Closing
#1013 on this PR would require either landing the `perf/1008` levers first or
re-scoping the issue to the panel evidence above.

No wall-clock claim is made anywhere in this section. Machine load reached 43.56
during capture (`mediaanalysisd` at 215%, `suggestd` at 96% — macOS indexing
triggered by the corpus reads). Every number above is a deterministic status,
iteration count, or counter.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01NKbF9EYSGFL1tnKYfoGqtg

---
_Generated by [Claude Code](https://claude.ai/code/session_01NKbF9EYSGFL1tnKYfoGqtg)_

