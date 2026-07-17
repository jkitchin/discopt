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
