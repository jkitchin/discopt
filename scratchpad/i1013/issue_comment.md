## The reproducer in this issue cannot be run as written on `main`

Working PR #1018 against this issue. Recording a premise shift before anyone
else spends time on the cells here.

### The five perturbation levers do not exist

`DISCOPT_LP_REFACTOR_INTERVAL`, `DISCOPT_LU_SYMBOLIC_REUSE`, and the LU
`pivot_threshold` knob return **zero hits crate-wide**. They were `perf/1008`
branch flags; that branch is unmerged, so the cells in this issue cannot be
constructed on `main` today. This is not a corpus-availability problem —
`QPLIB_0911` is present locally and I captured it (5150×1325, nnz=45146).

The three rounding-level knobs `main` *does* have
(`DISCOPT_LU_DENSITY_ROUTE`, `DISCOPT_LP_FACTORIZATION_HARDENING`,
`DISCOPT_LP_ITERATIVE_REFINEMENT`) are **bit-identical no-ops** on this LP —
same objective to 12 decimals, same 553 degenerate pivots, same max run — so
they are not a substitute arm. Result: inconclusive, not negative.

### The baseline reproduces this issue exactly

| | |
|---|---|
| pivots | 1279 |
| status | `optimal` |
| objective | −76.385275000002 |
| `DualDegeneratePivots` | **553 (43%)** |
| `DualStallTrips` | 0 |

### But the stall guard cannot fire here

`DualDegenerateRunMax = 40`, against #1018's patience of **2048**. Off by 51×,
and impossible in principle — the entire solve is 1279 pivots, so no run can
reach 2048. 8/8 cells across both arms: identical status, identical iteration
count, zero bails.

So the degeneracy this issue reports is real and reproducible, but it is
**short-run degeneracy** (longest run 40), not the long-stall pathology the
patience is set to catch. Those are different phenomena and the guard in #1018
addresses only the second.

### Panel context

`DualDegenerateRunMax` over 100 LPs, guard off: min 0, median **7**, p90 341,
max 7888. Distribution `{0: 2, 1–99: 78, 100–999: 16, 1000–2047: 1, ≥2048: 3}`.
`QPLIB_0911`'s 40 sits in the ordinary bulk. The 3 LPs above 2048 are exactly
the 3 where the guard fires; nothing lies between 1274 and 3352.

### What this means for this issue

The work item as phrased ("run the five cells, confirm the guard converts the
`iter_limit`") is not completable. Two ways forward:

1. **Land the `perf/1008` levers first**, then run the cells as written.
2. **Re-scope this issue** to what is measurable now: 43% degenerate pivots on a
   solve whose longest run is 40 is a *pivot-selection* question (short repeated
   cycling), not a *stall-bail* question. #1018's guard is the right fix for the
   long-run tail and does not claim to address this.

I have not picked between these — it is a scoping call for the owner. Full
data and driver in `scratchpad/i1013/cells_0911.{py,jsonl}` on PR #1018.

### Instrument defect, disclosed

The first run of these cells produced 32 clean records with every counter `0`,
including `DualWarmSolves` — impossible for a script whose only entry point is
the warm dual solve. The counters are gated on `DISCOPT_PROFILE`, which my
driver did not set. My own build-verification guard passed the whole way because
it checked the counter symbols *existed*, not that they *fired*. Fixed
(`DualWarmSolves >= 1` is now a hard refusal). The panel in #1018 is unaffected —
its own harness already sets the variable.

No wall-clock claim is made above; every figure is a deterministic status,
iteration count, or counter.
