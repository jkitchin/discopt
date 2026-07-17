# Issue #704 — H2 entry experiment: improver-LNS contingent is denominated in abstract cost, not wall time

**Date:** 2026-07-17
**Issue:** [#704](https://github.com/jkitchin/discopt/issues/704) (split from #282 round 2, `docs/dev/issue-282-syn-rsyn-diagnosis-2026-07-17.md` §R2-7)
**Regime:** entry experiment only — measure-only, no production/solver code changed.
**Verdict:** **H2 CONFIRMED.**

## Hypothesis (H2)

`_improver_allowed` on the NLP-BB path (`python/discopt/solver.py`, the second copy at
~10005–10705; the same closure exists on the McCormick path at ~6245–8605) charges *fixed
abstract cost units* against a node-proportional contingent and **never measures wall time**:

```python
_HEUR_COST = {"rins": 5.0, "lbranch": 10.0}
...
def _improver_allowed(cost: float) -> bool:
    if not _heur_budget_on or tree.incumbent() is None:
        return True                                   # (gap 2) ungated pre-incumbent
    _nodes = float(tree.stats().get("total_nodes", 0))
    _weight = _HEUR_SUCCESS_GAIN * (_heur_state["found"] + 1) / (_heur_state["calls"] + 1)
    _contingent = _HEUR_BUDGET_OFFSET + _HEUR_BUDGET_QUOT * _nodes * _weight
    return (_heur_state["cost"] + cost) <= _contingent
```

Two structural gaps claimed:

1. **No wall clock.** A `rins` call is charged 5.0 and a `local_branching` call 10.0
   regardless of whether it took 50 ms or 6 s. On syn/rsyn a single `local_branching` call is
   multiple seconds, so a handful of improver calls burns a large slice of a 60 s budget
   before a node-count-proportional contingent can throttle them.
2. **Bypassed before the first incumbent** (`tree.incumbent() is None → return True`).

## Kill criterion (stated before running)

H2 is **FALSIFIED** if any of: improver calls are uniformly short (≪ 1 s); OR total improver
wall time is a small fraction of the budget; OR the abstract cost already tracks wall time
proportionally (i.e. wall-seconds-per-charged-unit is roughly constant across families).
H2 is **CONFIRMED** if a single call is multiple seconds AND improvers consume a large slice
of the budget while the fixed abstract contingent stays unthrottled.

## Method

Measure-only. A local scratch harness (`scratchpad/h2_probe.py`, not committed) monkeypatches
`discopt._jax.primal_heuristics.rins` / `.local_branching` with timing wrappers (the solver
imports them at call time inside the node loop, so the module-attribute swap is picked up) and
uses a `node_callback` to timestamp the first incumbent. Instances loaded with `dm.from_nl`
from `~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/`, oracle from `minlplib.solu`,
`time_limit=60`, default (auto → NLP-BB) path. A separate `cProfile` run corroborates the
issue's original methodology. Raw data: `discopt_benchmarks/results/issue704/h2_walltime_2026-07-17.json`.

**Load caveat (#702).** Wall-clock numbers are soft under contention. These ran on a 14-core
box at load ≈ 3.3–3.7 with one concurrent CPU-bound `python` process (another agent). The
*direction and magnitude* of the finding (multi-second calls, 20–69 % of budget, 21× spread
in cost-per-unit) is far larger than plausible load noise, but the exact seconds are not
reproducible to the decimal. Single run per instance, no variance estimate.

## Results (time_limit = 60 s)

### Per-call wall time vs charged abstract cost

| instance | role | calls | total wall | mean / call | max / call | charged/call | wall ÷ charged (s per unit) |
|---|---|---|---|---|---|---|---|
| rsyn0805m | rins    | 12 | 18.87 s | 1.57 s | 3.18 s | 5.0 | **0.314** |
| rsyn0805m | lbranch |  5 | 20.26 s | 4.05 s | 4.44 s | 10.0 | 0.405 |
| rsyn0810m | rins    |  8 | 20.71 s | 2.59 s | 5.06 s | 5.0 | **0.518** |
| rsyn0810m | lbranch |  5 | 20.90 s | 4.18 s | 4.71 s | 10.0 | 0.418 |
| syn30hfsg | rins    | 13 |  2.34 s | 0.18 s | 0.29 s | 5.0 | **0.036** |
| syn30hfsg | lbranch |  4 | 10.20 s | 2.55 s | 2.72 s | 10.0 | 0.255 |
| syn40hfsg | rins    | 11 |  1.33 s | 0.12 s | 0.20 s | 5.0 | **0.024** |
| syn40hfsg | lbranch |  3 | 10.53 s | 3.51 s | 3.80 s | 10.0 | 0.351 |

The **cost-per-unit for `rins` varies 21×** (0.024 → 0.518 s/unit) across families while the
charged cost is a fixed 5.0 every time. `local_branching` is **2.5–4.7 s per call** on every
family — never anywhere near "cheap" — yet is charged a flat 10.0. If the abstract
denomination were a decent wall-time proxy, the last column would be roughly constant; it is
not. The relative weighting is also wrong on some families: `rins` is charged half of
`lbranch` (5 vs 10), but on `syn40hfsg` a `rins` call (0.12 s) is ~3 % of a `lbranch` call
(3.51 s), not half.

### Total improver wall time within the 60 s budget

| instance | improver total wall | % of 60 s budget | nodes | obj (oracle) | dual bound |
|---|---|---|---|---|---|
| rsyn0805m | **39.13 s** | **65.2 %** | 351 | 1116.5 (1296.1) | 1640.3 |
| rsyn0810m | **41.61 s** | **69.3 %** | 191 | 1548.6 (1721.4) | 2486.4 |
| syn30hfsg | 12.54 s | 20.9 % | 373 | 138.16 (138.16) | 1375.0 |
| syn40hfsg | 11.86 s | 19.8 % | 347 |  64.45 (67.71) | 1659.2 |

On the convex `rsyn` half the improvers eat **~two-thirds of the entire budget**. On the
`hfsg` pair — where the incumbent is already at/near the oracle (nothing left for improvers to
find) — they still burn ~20 %, dominated by `local_branching` (2.5–3.5 s/call). The contingent
never sees any of this: it is counting 5s and 10s.

### Ungated window (budget elapsed before the first incumbent)

| instance | first incumbent at | at node |
|---|---|---|
| rsyn0805m |  7.58 s | 3 |
| rsyn0810m | 11.14 s | 3 |
| syn30hfsg | 11.38 s | 3 |
| syn40hfsg | 17.63 s | 3 |

Reproduces the issue's "first incumbent ≈ 8.5 s on rsyn0805m" (measured 7.6 s).

### cProfile corroboration (rsyn0805m, 60 s — reproduces the issue's cited datum)

```
ncalls  tottime  cumtime  percall  function
     3    0.001   18.932    6.311  primal_heuristics.py:1610(local_branching)
     6    0.001   16.195    2.699  primal_heuristics.py:1338(rins)
```

The issue cited `rsyn0805m/60 s: 3 calls / 18.3 s` for `local_branching`; cProfile gives
3 calls / **18.93 s**, **6.31 s per call**. (Profiler overhead lowers node throughput, so the
profiled run reaches fewer improver windows than the direct-timer run — 3 vs 5 `lbranch`
calls — and inflates per-call time; the direct-timer table above is the faithful, lower
measurement.)

## Verdict: **H2 CONFIRMED**

Against the kill criterion:

- Calls are **not** uniformly short: `local_branching` is 2.5–4.7 s per call on every family
  and `rins` reaches 5.06 s on rsyn0810m. ✗ (not falsified)
- Improver wall is **not** a small fraction: 65–69 % of budget on rsyn, ~20 % on syn. ✗
- The abstract cost does **not** track wall time: cost-per-unit for a single role spans 21×
  while the charged cost is a fixed constant. ✗

All three falsification routes fail; both confirmation conditions hold. **Gap (1) is decisively
confirmed** — the contingent's denomination is decoupled from real wall cost, so it cannot
self-calibrate to families where a sub-MIP is seconds long.

**Honest scoping of gap (2).** The pre-incumbent bypass (`tree.incumbent() is None → return
True`) is real in the shared closure, but on the **NLP-BB path** the RINS / local-branching
block is already nested under `if _lns_inc is not None` (`solver.py:10589`), so those two
improvers cannot fire before the first incumbent regardless of the gate. The bypass therefore
does **not** open an extra improver window for RINS/lbranch on this path (it matters for the
McCormick-path `enumerate` heuristic at ~8279, not exercised here). The dominant, measured
defect is unambiguously gap (1). A time-denominated redesign should still gate the
pre-first-incumbent window for the improvers that *can* fire there, but the entry data does not
show gap (2) as an independent cost on the syn/rsyn NLP-BB runs.

## Proposed fix (design only — do NOT implement here; needs the Regime-2 panel gate)

Denominate the contingent in **measured wall seconds** against a fraction of remaining wall
budget, keeping the existing success-weighting so improvers that stop paying off still shut
down (SCIP `heur_subnlp` is time-aware in this spirit). Sketch, behind a flag, default-OFF:

- Time each improver call (`t = perf_counter()` around the `rins` / `local_branching` call)
  and charge the **measured seconds** to `_heur_state["cost"]` instead of `_HEUR_COST[...]`.
- Express the contingent as a fraction of *remaining* budget, e.g.
  `contingent = HEUR_TIME_FRAC * max(0, deadline - now) * _weight` with the same
  `3·(found+1)/(calls+1)` success weight, so a family whose sub-MIPs cost seconds is throttled
  by the same rule that lets a family with 50 ms sub-MIPs run many calls.
- Also gate the pre-first-incumbent window for any improver that can fire there (close gap 2
  for the McCormick `enumerate` path), so early cheap-looking abstract costs don't front-load.

Soundness is not at risk either way: B&B stays exhaustive, so a throttled improver costs nodes,
never a wrong optimum. Graduation must clear the CLAUDE.md Regime-2 panel gate (flag
default-OFF; `incorrect_count = 0`; no bound above oracle; no `gap_certified=True` → uncertified
regression; incumbents independently feasibility-verified; net-positive corpus-wide), and must
be A/B'd on the families that motivated the governor (#347 `clay*`, #321) so the fix does not
re-break the instances the improvers were built for. Per §R2-7, none of this moves a default
until that gate passes.

## Artifacts

- Raw JSON: `discopt_benchmarks/results/issue704/h2_walltime_2026-07-17.json`
- Harness (not committed): `scratchpad/h2_probe.py`, `scratchpad/h2_cprofile.py`

---

## Build result (2026-07-17): the fix is FALSIFIED — net-negative. #704 closed.

The wall-time contingent was implemented exactly as proposed above (`_improver_gate_allows`
pure helper shared by both LNS loops; `DISCOPT_HEUR_WALLTIME` charges the call's measured wall
duration against `HEUR_TIME_FRAC` of the remaining budget; default-OFF). It is sound (adversarial
suite 10/10; the improvers only propose feasibility-verified incumbents and B&B stays exhaustive)
and the flag-OFF path is byte-identical.

But the differential panel **falsifies H2's implied lever**. Flag OFF vs ON, syn/rsyn, TL=60 s,
quiet single-threaded box (all four runs sound — incumbent ≤ oracle ≤ bound):

| instance | opt | OFF obj / bound | ON obj / bound | reported gap OFF→ON |
|---|---|---|---|---|
| rsyn0805m | 1296.1 | 1116.5 / 1631.9 | 1055.9 / 1568.8 | 46.2 % → **48.6 %** (looser) |
| rsyn0810m | 1721.4 | 1548.6 / 2464.7 | 1417.3 / 2358.0 | 59.2 % → **66.4 %** (looser) |
| rsyn0815m | 1269.9 | 1047.3 / 1877.5 | 1041.5 / 1836.3 | 79.3 % → 76.3 % (tighter) |
| syn40m | 67.7 | 33.2 / 1228.7 | 23.2 / 1158.7 | 3601 % → **4903 %** (looser) |

Wall-mode does exactly what H2 predicted — it throttles the seconds-long improvers, and the freed
budget **tightens the dual bound** (rsyn0805m 1631.9 → 1568.8). But it **worsens the incumbent by a
near-equal amount** (1116.5 → 1055.9): the improvers were *buying primal quality worth more than the
dual gain*. The trade is ~1:1-to-unfavorable, so the reported gap does not shrink — it **loosens on
3/4**. The premise that the improvers waste budget is wrong; they are productive, and reallocating
their budget to the search just moves the gap from the primal side to the dual side (net worse).

**Resolution.** H2 (the calibration *is* decoupled from wall time) was confirmed (§ above, PR #709),
but its implied *fix* — throttle improvers to free budget — is **not a lever**: net-negative on the
target family. Per CLAUDE.md §4 the measurement wins, and per "sound ≠ helpful" a net-negative change
does not ship even default-OFF. The code is reverted; this record stands. The real #282 lever is
**dual-bound quality** (the root relaxation), pursued in #715 — not primal-budget reallocation.
Closing #704.
