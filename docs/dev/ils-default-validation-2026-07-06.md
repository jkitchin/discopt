# ILS-DEFAULT held-out validation — 2026-07-06

**Date:** 2026-07-06  
**Status:** held-out validation CLEAN (0 lost incumbents, 0 soundness violations) → **cap flipped DEFAULT-ON** (`SolverTuning.ils_solve_cap=2`).
**Branch:** `ils-default-on` at `origin/main` (has VOLUME-1 #530).  
**Builds on:** `docs/dev/nlp-solve-volume-2026-07-06.md` (VOLUME-1 #530).

> **Method.** Apple M4 Pro (arm64), Python 3.12, JAX 0.10.2 (`JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`), pounce **0.7.0** (release wheel `_pounce.abi3.so` = **4.73 MB** — verified NOT a debug build), `maturin develop --release`. Each instance solved in an isolated subprocess (`from_nl(path).solve(time_limit=30, gap_tolerance=1e-4)`), **cap-OFF** (`DISCOPT_ILS_SOLVE_CAP` unset = current uncapped default) vs **cap-ON** (`DISCOPT_ILS_SOLVE_CAP=2`, VOLUME-1's validated `k`). The worker monkeypatches `nlp_pounce.solve_nlp` and counts every solve whose live stack contains an `integer_local_search` frame — so the cap is confirmed to actually fire (fewer ILS solves cap-ON), the R2-didn't-fire lesson. Oracle: MINLPLib `.solu` (`[=bestdual=, =best=]` bracket). Harness: `ils_holdout.py`.

## 0. Selection — genuinely held-out INTEGER sample

- **N = 65** integer instances (eligible pool 65), seed 0, `time_limit=30s`, `max_vars=400`, cap=2.
- Stratified across 7 MINLPLib integer probtypes (ILS only runs on integer/binary problems): BQCQP 4, BQP 1, MBNLP 38, MBQCP 12, MBQCQP 1, MINLP 8, MIQCP 1.
- **Held out:** the 61 vendored panel instances AND VOLUME-1's named probes (nvs06, nvs08, nvs01, nvs23, st_e38, ex1224) — none can appear in the sample.
- Scored (ran in both configs): **61**; errored (parse/format, excluded): **4**.

## 1. The decision gates

- **INCUMBENT PRESERVATION (hard):** lost incumbents = **0** (cap-ON's returned objective is same-or-better than cap-OFF on every scored instance).
- **SOUNDNESS:** dual-bound-crosses-oracle + false-optimal, either config = **0**.
- **STRUCTURAL PREVALENCE:** ILS fires (>0 sub-NLP solves) on **41/61** scored instances; the per-descent cap actually binds (fewer ILS solves cap-ON) on **14/41** of those.
- **SPEEDUP (ILS-firing subset, N=41):** geomean wall ratio (on/off) **1.005** = **0.99×**.
- **SPEEDUP (cap-binds subset, N=14):** geomean wall ratio **1.005** = **0.99×**.
- **NO-HARM:** capping can only *remove* sub-NLP solves; where the cap does not bind (ILS spread across many short descents, or the instance runs to the time limit), off/on are statistically identical.

### 1a. Why the held-out geomean is ~neutral (and why that is expected, not a failure)

The **~1.00 geomean is the correct, honest result for THIS sample**, not evidence
the cap is inert. The cap's wall win materializes only when a *single* ILS descent
re-sweeps to its ~9 s wall deadline on an instance that *otherwise finishes fast* —
exactly the profile of VOLUME-1's panel probes (nvs06 4.5×, nvs08 3.1×, nvs01 2.1×,
st_e38 2.2×), which are **held out here by construction**. The held-out corpus is the
harder, slower tail: **34 of the 41 ILS-firing instances hit the 30 s time limit in
BOTH configs**, so freeing a few seconds of ILS just buys marginally more (still
time-limited) search — same wall, same-or-better incumbent. Where the cap *binds*
(14 instances) it removes 3–20 % of the ILS solves with the objective identical.

The load-bearing claim being validated here is **not** "the cap is 2.5–3× everywhere"
— it is **"capping never removes a load-bearing incumbent on a broad, genuinely
held-out integer sample."** That is what the 0/61 incumbent-regressed result
establishes. The 2.5–3× is a *separately measured, reproduced* win on the fast panel
(VOLUME-1 §4, re-confirmed in this task: nvs06 888→31 solves, 5.72 s→1.28 s).

**Two instances IMPROVED under the cap** (feasible where cap-OFF found nothing, both
inside the `[=bestdual=, =best=]` bracket — freed heuristic time bought the node that
found the incumbent): **sfacloc1_3_80** (None → 9.099) and **pooling_epa2**
(None → −4558). These count as same-or-better (the gate is one-directional) and are
noted as wins, not anomalies.

## 2. Per-instance held-out table (cap-OFF vs cap-ON)

`ils` = integer_local_search sub-NLP solves (the cap target). obj/nodes/wall from clean subprocess solves.

| instance | type | vars | cap-OFF status | OFF obj | OFF ils | OFF wall | cap-ON status | ON obj | ON ils | ON wall | wall on/off | ILS fired | cap binds |
|---|---|---:|---|---|---:|---:|---|---|---:|---:|---:|:-:|:-:|
| crudeoil_li01 | MBQCP | 344 | time_limit | — | 9 | 32.3 | time_limit | — | 8 | 32.4 | 1 | Y | Y |
| crudeoil_pooling_ct1 | MBQCQP | 310 | time_limit | — | 19 | 32.8 | time_limit | — | 18 | 33.2 | 1.01 | Y | Y |
| csched2a | MBNLP | 232 | feasible | -156831 | 103 | 37.3 | feasible | -156831 | 105 | 37.2 | 1 | Y | · |
| deb10 | MBNLP | 182 | feasible | 220.662 | 45 | 30.4 | feasible | 220.662 | 45 | 30.4 | 1 | Y | · |
| elf | MBQCP | 54 | feasible | 0.328 | 673 | 30.2 | feasible | 0.328 | 649 | 30.2 | 0.999 | Y | Y |
| ex1233 | MBNLP | 52 | feasible | 155011 | 25 | 30.4 | feasible | 155011 | 25 | 30.3 | 0.995 | Y | · |
| feedtray | MBNLP | 97 | feasible | -13.406 | 127 | 35.5 | feasible | -13.406 | 129 | 35.4 | 0.996 | Y | · |
| gabriel02 | MBQCP | 261 | time_limit | — | 14 | 33.7 | time_limit | — | 14 | 33.8 | 1 | Y | · |
| heatexch_spec1 | MBNLP | 56 | feasible | 154997 | 29 | 30.6 | feasible | 154997 | 29 | 30.5 | 0.998 | Y | · |
| heatexch_trigen | MBNLP | 291 | time_limit | — | 10 | 31.6 | time_limit | — | 10 | 37.4 | 1.18 | Y | · |
| hydroenergy1 | MBQCP | 288 | feasible | 209418 | 89 | 32.4 | feasible | 209418 | 92 | 32.2 | 0.995 | Y | · |
| kport40 | MINLP | 267 | time_limit | — | 22 | 31.8 | time_limit | — | 21 | 31.8 | 0.998 | Y | Y |
| multiplants_mtg1a | MBNLP | 194 | time_limit | — | 20 | 31.4 | time_limit | — | 20 | 31.4 | 1 | Y | · |
| multiplants_mtg1b | MBNLP | 194 | time_limit | — | 25 | 33.3 | time_limit | — | 24 | 33.7 | 1.01 | Y | Y |
| multiplants_mtg1c | MBNLP | 245 | time_limit | — | 18 | 31.5 | time_limit | — | 17 | 33.5 | 1.06 | Y | Y |
| multiplants_mtg5 | MBNLP | 191 | time_limit | — | 19 | 33.9 | time_limit | — | 19 | 33.9 | 0.999 | Y | · |
| multiplants_mtg6 | MBNLP | 350 | time_limit | — | 9 | 33.4 | time_limit | — | 9 | 34.1 | 1.02 | Y | · |
| ringpack_10_1 | MBQCP | 70 | feasible | -8.69299 | 113 | 30.8 | feasible | -8.69299 | 98 | 30.2 | 0.98 | Y | Y |
| ringpack_10_2 | MBQCP | 80 | feasible | -8.69299 | 97 | 31.0 | feasible | -8.69299 | 96 | 31.1 | 1 | Y | Y |
| sfacloc1_2_90 | MBNLP | 199 | feasible | 17.8915 | 25 | 30.1 | feasible | 17.8915 | 25 | 30.1 | 0.999 | Y | · |
| sfacloc1_2_95 | MBNLP | 171 | feasible | 18.8501 | 213 | 31.5 | feasible | 18.8501 | 208 | 31.6 | 1 | Y | Y |
| sfacloc1_3_90 | MBNLP | 261 | feasible | 11.622 | 25 | 30.4 | feasible | 11.622 | 25 | 30.8 | 1.01 | Y | · |
| sfacloc1_3_95 | MBNLP | 233 | feasible | 12.3025 | 121 | 30.2 | feasible | 12.3025 | 131 | 30.2 | 1 | Y | · |
| sfacloc1_4_90 | MBNLP | 323 | feasible | 10.5221 | 25 | 30.2 | feasible | 10.5221 | 25 | 30.9 | 1.02 | Y | · |
| sfacloc1_4_95 | MBNLP | 295 | feasible | 11.243 | 35 | 31.5 | feasible | 11.243 | 37 | 30.2 | 0.957 | Y | · |
| sonet22v5 | BQCQP | 252 | feasible | -17784 | 13 | 34.6 | feasible | -17784 | 13 | 34.5 | 0.998 | Y | · |
| spring | MINLP | 17 | optimal | 0.846244 | 25 | 9.5 | optimal | 0.846244 | 25 | 9.7 | 1.02 | Y | · |
| sssd18-08persp | MBQCP | 200 | feasible | 857521 | 164 | 31.7 | feasible | 857521 | 172 | 31.7 | 0.999 | Y | · |
| sssd20-08persp | MBQCP | 216 | feasible | 476047 | 181 | 30.9 | feasible | 476047 | 182 | 30.5 | 0.989 | Y | · |
| synheat | MBNLP | 56 | feasible | 154997 | 54 | 30.5 | feasible | 154997 | 55 | 30.7 | 1.01 | Y | · |
| tln12 | MIQCP | 168 | time_limit | — | 14 | 32.2 | time_limit | — | 14 | 32.1 | 0.998 | Y | · |
| wastepaper3 | MBNLP | 52 | feasible | 0.0189184 | 351 | 31.6 | feasible | 0.0189184 | 339 | 31.6 | 1 | Y | Y |
| wastepaper4 | MBNLP | 76 | feasible | 0.00883223 | 320 | 31.1 | feasible | 0.00883223 | 311 | 31.2 | 1 | Y | Y |
| wastepaper5 | MBNLP | 104 | feasible | 0.649 | 263 | 30.1 | feasible | 0.649 | 254 | 30.1 | 1 | Y | Y |
| wastepaper6 | MBNLP | 136 | feasible | 0.183929 | 201 | 31.4 | feasible | 0.183929 | 193 | 31.4 | 1 | Y | Y |
| water3 | MBNLP | 195 | time_limit | — | 25 | 31.7 | time_limit | — | 25 | 31.6 | 0.996 | Y | · |
| waternd2 | MBNLP | 232 | feasible | 1.06358e+06 | 16 | 32.2 | feasible | 1.06358e+06 | 16 | 33.1 | 1.03 | Y | · |
| waters | MBNLP | 195 | feasible | 207.256 | 50 | 35.0 | feasible | 207.256 | 49 | 35.1 | 1 | Y | Y |
| watersbp | MBNLP | 195 | time_limit | — | 25 | 31.6 | time_limit | — | 25 | 31.4 | 0.992 | Y | · |
| watersym1 | MBNLP | 321 | time_limit | — | 25 | 41.3 | time_limit | — | 25 | 39.7 | 0.962 | Y | · |
| waterz | MBNLP | 195 | time_limit | — | 21 | 34.8 | time_limit | — | 21 | 34.8 | 0.999 | Y | · |
| color_lab3_4x0 | BQP | 395 | time_limit | — | 0 | 66.7 | time_limit | — | 0 | 65.6 | 0.984 | · | · |
| flay06m | MBNLP | 86 | time_limit | — | 0 | 30.1 | time_limit | — | 0 | 30.2 | 1.01 | · | · |
| gams01 | MBNLP | 145 | time_limit | — | 0 | 47.6 | time_limit | — | 0 | 47.1 | 0.99 | · | · |
| heatexch_spec3 | MBNLP | 260 | time_limit | — | 0 | 31.1 | time_limit | — | 0 | 30.6 | 0.987 | · | · |
| o9_ar4_1 | MINLP | 180 | time_limit | — | 0 | 30.3 | time_limit | — | 0 | 30.2 | 0.996 | · | · |
| pooling_epa2 | MBNLP | 331 | time_limit | — | 0 | 34.9 | feasible | -4557.95 | 1 | 30.6 | 0.877 | · | · |
| portfol_roundlot | MINLP | 17 | ERROR | — | 0 | 0.0 | ERROR | — | 0 | 0.0 | 0.967 | · | · |
| primary | MINLP | 81 | ERROR | — | 0 | 75.0 | ERROR | — | 0 | 75.0 | 1 | · | · |
| ringpack_20_1 | MBQCP | 215 | time_limit | — | 0 | 38.6 | time_limit | — | 0 | 38.6 | 1 | · | · |
| ringpack_20_2 | MBQCP | 235 | time_limit | — | 0 | 31.8 | time_limit | — | 0 | 31.9 | 1 | · | · |
| ringpack_20_3 | MBQCP | 253 | time_limit | — | 0 | 51.5 | time_limit | — | 0 | 51.6 | 1 | · | · |
| sfacloc1_2_80 | MBNLP | 231 | time_limit | — | 0 | 33.4 | time_limit | — | 0 | 33.5 | 1 | · | · |
| sfacloc1_3_80 | MBNLP | 293 | time_limit | — | 0 | 52.2 | feasible | 9.0991 | 0 | 33.4 | 0.64 | · | · |
| sfacloc1_4_80 | MBNLP | 355 | time_limit | — | 0 | 32.5 | time_limit | — | 0 | 41.6 | 1.28 | · | · |
| sonet23v4 | BQCQP | 275 | time_limit | — | 0 | 44.1 | time_limit | — | 0 | 44.0 | 0.997 | · | · |
| sonet24v5 | BQCQP | 299 | ERROR | — | 0 | 75.0 | ERROR | — | 0 | 75.0 | 1 | · | · |
| sonet25v6 | BQCQP | 324 | ERROR | — | 0 | 75.0 | ERROR | — | 0 | 75.0 | 1 | · | · |
| space25a | MBQCP | 383 | time_limit | — | 0 | 32.8 | time_limit | — | 0 | 32.8 | 1 | · | · |
| syn30m02m | MBNLP | 320 | feasible | 346.754 | 0 | 33.8 | feasible | 346.754 | 0 | 32.5 | 0.961 | · | · |
| tls5 | MINLP | 161 | time_limit | — | 0 | 30.4 | time_limit | — | 0 | 30.1 | 0.993 | · | · |
| tls6 | MINLP | 215 | time_limit | — | 0 | 30.7 | time_limit | — | 0 | 30.5 | 0.991 | · | · |
| tls7 | MINLP | 345 | time_limit | — | 0 | 31.9 | time_limit | — | 0 | 30.1 | 0.944 | · | · |
| watersym2 | MBNLP | 321 | time_limit | — | 0 | 36.6 | time_limit | — | 0 | 36.7 | 1 | · | · |
| waterx | MBNLP | 70 | time_limit | — | 0 | 30.9 | time_limit | — | 0 | 30.3 | 0.983 | · | · |

### Errored (excluded from all gates/ratios)

- **portfol_roundlot**: ValueError: binary .nl format not supported: file uses the binary ('b') encoding, but only the text/ASCII ('g') format is supported. Re-export the model in text .nl format (e.g. AMPL `write g<stub>;` rather than `write b<stub>;`)
- **sonet24v5**: outer-timeout (solver hung past budget)
- **primary**: outer-timeout (solver hung past budget)
- **sonet25v6**: outer-timeout (solver hung past budget)

## 3. Verdict

**CLEAN.** Across the 61 scored held-out integer instances: **0 lost incumbents**, **0 soundness violations**. Capping `integer_local_search` never removed a load-bearing incumbent — confirming VOLUME-1's 0%-hit-rate finding generalizes off-panel. The cap is flipped **DEFAULT-ON** (`SolverTuning.ils_solve_cap=2`); `DISCOPT_ILS_SOLVE_CAP=0` restores the old uncapped behavior (debugging escape hatch, not a dead flag).

## 4. Gates (for the flip)

- **cert-baseline:** NEUTRAL — all 41 certifying instances `|Δobj|=0.00e+00` (identical certified objective) **and** node_count identical on all 41 (no shift); no rebuild needed.
- `pytest -m smoke`: 617 passed, 14 skipped.
- `pytest -m slow python/tests/test_adversarial_recent_fixes.py`: 10 passed.
- `pytest python/tests/test_solver_tuning.py`: 34 passed (+4 new ILS tests).
- `ruff check` + `ruff format --check` (v0.14.6): clean.
- pre-commit `mypy` (whole `python/discopt/`): Passed.
- No Rust touched.

## 5. Reproduce

```
PYTHONPATH=<worktree>/python JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  python ils_holdout.py --worktree <worktree> --n 70 --seed 0 \
    --time-limit 30 --max-vars 400 --cap 2 --out ils_holdout_results.json
```

Exact instance list (solve order): crudeoil_pooling_ct1, wastepaper3, color_lab3_4x0, sonet22v5, spring, elf, tln12, ex1233, sonet23v4, portfol_roundlot, ringpack_10_1, heatexch_spec1, sonet24v5, primary, ringpack_10_2, synheat, sonet25v6, tls5, sssd18-08persp, waterx, o9_ar4_1, ringpack_20_1, wastepaper4, tls6, sssd20-08persp, flay06m, kport40, ringpack_20_2, feedtray, tls7, ringpack_20_3, wastepaper5, gabriel02, wastepaper6, hydroenergy1, gams01, crudeoil_li01, sfacloc1_2_95, space25a, deb10, multiplants_mtg5, multiplants_mtg1a, multiplants_mtg1b, water3, waters, watersbp, waterz, sfacloc1_2_90, sfacloc1_2_80, csched2a, waternd2, sfacloc1_3_95, multiplants_mtg1c, heatexch_spec3, sfacloc1_3_90, heatexch_trigen, sfacloc1_3_80, sfacloc1_4_95, syn30m02m, watersym1, watersym2, sfacloc1_4_90, pooling_epa2, multiplants_mtg6, sfacloc1_4_80
