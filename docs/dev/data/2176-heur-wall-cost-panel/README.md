# #2176: the improver contingent priced by measured wall (2026-11-06)

CLAUDE.md §5 graduation panel for `DISCOPT_HEUR_WALL_COST`, measured on `f32b4d3b`
(content-identical to `bcbb5676`), flag ON vs OFF, interleaved per instance. With the
flag ON, the #1344 contingent keeps its formula and measures both sides:

- one unit is the loop's node-processing wall (`rust_time + jax_time`) per node;
- a call costs its heuristic's mean measured wall;
- a first call keeps the fixed-cost test.

Host: 4 cores. Load 1-min was 1-8.

## Corpus: `corpus_60s_2rep.jsonl`, `corpus_report.txt`

Command: `flag_onoff_corpus_panel.py --flag DISCOPT_HEUR_WALL_COST --solu minlplib.solu
--time-limit 60 --jobs 2 --reps 2` over the 74 in-repo `.nl`. The oracle is
`www.minlplib.org/minlplib.solu`, fetched for this run.

- 148 rows, 0 problem instances.
- Certified 116 -> 116; gained none, lost none.
- 260 incumbent feasibility checks, 0 errors.
- Total wall 2466 -> 2449 s; SGM ratio (shift 1 s) 0.990, sd of log 0.041.
- 9 of 148 rows have different node counts, in both directions, at noise level.

On this corpus the improvers rarely run long enough for pricing to matter.

## gdp_parity: `gdp_parity_120s.jsonl`, `gdp_report.txt`

Command: `issue1917_gdp_flag_panel.py --flag DISCOPT_HEUR_WALL_COST --phase solve --all
--reps 1 --time-limit 120`. The run split into two processes over disjoint instance lists,
and was resumed once after a container restart (the driver skips finished rows).
mod_hens and kaibel use the #2170 harness shim for their builders' `ipopt` calls, in both
arms.

- 96 solves, incorrect 0, bound crossings 0.
- SGM wall (shift 10 s, cap 120 s) 37.42 -> 36.71 s.
- The report flags one loss, **syngas hull**: OFF optimal at 119.2 s against a 120 s
  limit, ON no LOA incumbent by 120.1 s, 3 nodes in both arms. Re-run alone,
  interleaved, 3 reps per arm (`syngas_rerun.py`, `syngas_hull_rerun.jsonl`), it was
  optimal in all 6 solves: OFF 112.8 / 113.8 / 113.5 s, ON 112.7 / 113.0 / 113.3 s.
  The wall-priced branch was taken **0** times in every run, because no improver runs
  before LOA's incumbent, which arrives at ~110 s. The flag cannot act on this cell; the
  panel row is time-limit noise at panel load.

Where it acts:

| cell | OFF | ON |
|---|---|---|
| positioning big-M | 151 nodes, 9.9-10.1 s | 305 nodes, 7.0-7.1 s |
| positioning hull | 151 nodes, 9.7-9.9 s | 305 nodes, 6.8-7.0 s |
| batch_processing big-M | 503 nodes, bound 557177.08 | 885 nodes, bound 566697.08 |
| batch_processing hull | 255 nodes, bound 588573.36 | 541 nodes, bound 602199.55 |

positioning is optimal (-8.064136108923352) in every arm. Its walls are 3 interleaved reps
(`positioning_rerun.py`) and deterministic in nodes. batch_processing is not certified in
either arm. Its incumbent is the same under big-M (1034233.32) and run-to-run variable
under hull (OFF 1019222.46, ON 1050062.12 here; the entry experiment had OFF 1623421, ON
1537635).

## Verdict

**Graduated** (default ON, `=0` restores the fixed-cost test).

- **Cert-clean:** 0 incorrect, 0 bound crossings, 0 certificates lost once the syngas
  noise row is accounted for.
- **Net-positive** on the class it targets, where an improver call costs many nodes'
  worth of wall (positioning -30 % wall, batch_processing bound +1.7 % / +2.3 %);
  neutral elsewhere.
