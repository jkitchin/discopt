# baron_global50 head-to-head re-run — current `main` (2026-07-07)

Re-run of the 50-instance curated global-opt head-to-head (`discopt_benchmarks/config/baron_global50.txt`)
against full-license BARON (via GAMS 53), on the current solver
(`origin/main` @ `d1b09d3e`, all this session's wins default-ON: ILS cap, PSD
gate, zero-spanning lift, effort governor, plus the C-38/C-40 correctness fixes).

- **Solver build:** `maturin develop --release` (pounce `_pounce.abi3.so` = 4.5 MB,
  confirmed a release build — debug timings would be meaningless).
- **Harness:** `discopt_benchmarks/scripts/global_opt_baron_vs_discopt.py`,
  `--time-limit 60`, both solvers fresh (no reuse). BARON driven the established
  way: `gams <name>.gms minlp=baron optcr=0 optca=1e-9 reslim=60`.
- **Correctness tol:** abs=1e-6, rel=1e-4. Oracle: MINLPLib `primalbound`
  (harness) **and** an independent cross-check against
  `~/Dropbox/projects/discopt-minlp-benchmark/minlplib.solu`
  (`=opt=`/`=best=` primal, `=bestdual=` dual).
- **Raw JSON:** `docs/dev/global50-rerun-2026-07-07.json`.

## HARD CORRECTNESS GATE — PASS (0 violations)

Independent cross-check of every discopt certified-optimal incumbent and every
discopt dual bound against `minlplib.solu`:

- **discopt VIOLATIONS: 0** — no false-optimal, no incumbent strictly better than
  the proven global, no dual bound crossing the oracle.
- **BARON VIOLATIONS: 0.**

This confirms the C-38/C-40 fixes hold on the global50 set specifically (the
session's 250-instance sweep was clean; this is the confirmation on this panel).

## Headline

| metric | discopt (now) | discopt (prior 06-18) | BARON (fresh) |
|---|---|---|---|
| proved-optimal / 50 | **43** | 0 | 45 |
| total wall (s) | 476.6 | — | 130.9 |
| geomean wall (s, all 50) | 1.037 | — | 0.068 |

The story is the **discopt delta: 0 → 43 proved-optimal / 50**. The prior 06-18
run (baron 31/50, discopt 0/50) predates every relaxation/bounding/correctness
change in this session; discopt now certifies global optimality on 43 of the 50.

> "proved-optimal" = a *certified* global (discopt status `optimal`; BARON model
> status `1 Optimal`). BARON's `2 Locally Optimal` and `8 Integer Solution` are
> **not** certified global and do not count.

## The gap vs BARON

- **Jointly proved-optimal: 40.** On those 40, geomean wall: discopt 0.645 s vs
  BARON 0.049 s → **discopt ≈ 13.3× slower** on the jointly-proved set. BARON is
  still much faster per-instance; the gap is bounding/convergence speed, not
  correctness.
- **discopt-only proved (3):** `chance`, `dispatch`, `tspn05`. discopt certifies
  the global here while BARON stops at a *locally* optimal / integer-solution
  status (uncertified) within the 60 s budget.
- **BARON-only proved (5):** `nvs05`, `nvs09`, `tanksize`, `tls2`, `st_miqp5`.
  - `nvs05`, `nvs09`, `tanksize`, `tls2`: discopt found the **correct incumbent**
    (matches the oracle) but exhausted the 60 s budget without closing the gap —
    honest convergence gaps, not correctness failures. Dual bounds are valid
    (never cross the oracle).
  - `st_miqp5`: **data artifact, not a solver failure.** This instance is not in
    the vendored `python/tests/data/minlplib_nl/` corpus; the copy drawn from the
    full MINLPLib snapshot is in **binary `.nl`** format (`b` encoding), which the
    parser correctly refuses with a loud error (only text/`g` `.nl` is supported).
- **Two honest time-limits with no incumbent:** `casctanks`, `hda` — discopt
  returned no feasible point in 60 s (BARON: `casctanks` `8 Integer Solution`
  uncertified, `hda` `19 Infeasible` — an intermediate/relaxation status).
  `casctanks`'s discopt dual bound (-90.18) does **not** cross the oracle (9.16) —
  a valid lower bound.

## Per-instance results

| instance | solu best | discopt obj | discopt status | d wall (s) | BARON obj | BARON status | b wall (s) |
|---|---|---|---|---|---|---|---|
| alan | 2.925 | 2.925 | optimal | 0.2 | 2.925 | 1 Optimal | 0.1 |
| casctanks | 9.1635 | — | time_limit | 67.1 | 9.1635 | 8 Integer Solution | 61.0 |
| chance | 29.894 | 29.894 | optimal | 0.4 | 29.894 | 2 Locally Optimal | 0.0 |
| clay0303hfsg | 26669 | 26669 | optimal | 22.7 | 26669 | 1 Optimal | 1.4 |
| cvxnonsep_nsig30 | 130.63 | 130.63 | optimal | 2.6 | 130.63 | 1 Optimal | 1.9 |
| cvxnonsep_psig30 | 78.999 | 78.999 | optimal | 1.2 | 78.999 | 1 Optimal | 0.8 |
| cvxnonsep_psig40r | 86.545 | 86.545 | optimal | 3.4 | 86.545 | 1 Optimal | 0.4 |
| dispatch | 3155.3 | 3155.3 | optimal | 0.4 | 3155.3 | 2 Locally Optimal | 0.0 |
| ex1221 | 7.6672 | 7.6672 | optimal | 0.5 | 7.6672 | 1 Optimal | 0.0 |
| ex1222 | 1.0765 | 1.0765 | optimal | 0.4 | 1.0765 | 1 Optimal | 0.0 |
| ex1224 | -0.94347 | -0.94347 | optimal | 2.1 | -0.9435 | 1 Optimal | 0.1 |
| ex1225 | 31 | 31 | optimal | 1.1 | 31 | 1 Optimal | 0.0 |
| ex1226 | -17 | -17 | optimal | 0.5 | -17 | 1 Optimal | 0.0 |
| fac2 | 3.3184e+08 | 3.3184e+08 | optimal | 5.6 | 3.3184e+08 | 1 Optimal | 0.0 |
| flay02m | 37.947 | 37.947 | optimal | 0.5 | 37.947 | 1 Optimal | 0.0 |
| flay03m | 48.99 | 48.99 | optimal | 6.6 | 48.99 | 1 Optimal | 0.3 |
| gbd | 2.2 | 2.2 | optimal | 0.1 | 2.2 | 1 Optimal | 0.0 |
| gkocis | -1.9231 | -1.9231 | optimal | 0.7 | -1.9231 | 1 Optimal | 0.0 |
| hda | -5964.5 | — | time_limit | 64.0 | — | 19 Infeasible | 0.2 |
| m3 | 37.8 | 37.8 | optimal | 3.3 | 37.8 | 1 Optimal | 0.0 |
| nvs01 | 12.47 | 12.47 | optimal | 2.8 | 12.47 | 1 Optimal | 0.1 |
| nvs02 | 5.9642 | 5.9642 | optimal | 0.1 | 5.9642 | 1 Optimal | 0.0 |
| nvs03 | 16 | 16 | optimal | 0.4 | 16 | 1 Optimal | 0.0 |
| nvs04 | 0.72 | 0.72 | optimal | 0.5 | 0.72 | 1 Optimal | 0.0 |
| nvs05 | 5.4709 | 5.4709 | feasible | 60.1 | 5.4709 | 1 Optimal | 0.5 |
| nvs06 | 1.7703 | 1.7703 | optimal | 1.4 | 1.7703 | 1 Optimal | 0.0 |
| nvs07 | 4 | 4 | optimal | 0.0 | 4 | 1 Optimal | 0.0 |
| nvs08 | 23.45 | 23.45 | optimal | 0.9 | 23.45 | 1 Optimal | 0.1 |
| nvs09 | -43.134 | -43.134 | feasible | 60.3 | -43.134 | 1 Optimal | 0.0 |
| nvs10 | -310.8 | -310.8 | optimal | 0.1 | -310.8 | 1 Optimal | 0.0 |
| nvs11 | -431 | -431 | optimal | 0.4 | -431 | 1 Optimal | 0.0 |
| nvs12 | -481.2 | -481.2 | optimal | 1.5 | -481.2 | 1 Optimal | 0.1 |
| nvs13 | -585.2 | -585.2 | optimal | 2.0 | -585.2 | 1 Optimal | 0.1 |
| nvs14 | -40358 | -40358 | optimal | 0.2 | -40358 | 1 Optimal | 0.0 |
| nvs15 | 1 | 1 | optimal | 0.1 | 1 | 1 Optimal | 0.0 |
| oaer | -1.9231 | -1.9231 | optimal | 0.7 | -1.9231 | 1 Optimal | 0.0 |
| st_e13 | 2 | 2 | optimal | 0.4 | 2 | 1 Optimal | 0.0 |
| st_e29 | -0.94347 | -0.94347 | optimal | 2.1 | -0.9435 | 1 Optimal | 0.0 |
| st_e36 | -246 | -246 | optimal | 16.5 | -246 | 1 Optimal | 0.1 |
| st_e38 | 7197.7 | 7197.7 | optimal | 0.6 | 7197.7 | 1 Optimal | 0.0 |
| st_miqp1 | 281 | 281 | optimal | 0.1 | 281 | 1 Optimal | 0.0 |
| st_miqp2 | 2 | 2 | optimal | 0.2 | 2 | 1 Optimal | 0.0 |
| st_miqp3 | -6 | -6 | optimal | 0.1 | -6 | 1 Optimal | 0.0 |
| st_miqp4 | -4574 | -4574 | optimal | 0.1 | -4574 | 1 Optimal | 0.0 |
| st_miqp5 | -333.89 | ERROR (binary .nl) | ERROR | 0.0 | -333.89 | 1 Optimal | 0.0 |
| st_test1 | 0 | -1.6496e-08 | optimal | 0.1 | 0 | 1 Optimal | 0.0 |
| st_testgr3 | -20.59 | -20.59 | optimal | 0.2 | -20.59 | 1 Optimal | 0.0 |
| tanksize | 1.2686 | 1.2686 | feasible | 60.1 | 1.2686 | 1 Optimal | 2.9 |
| tls2 | 5.3 | 5.3 | feasible | 60.0 | 5.3 | 1 Optimal | 0.0 |
| tspn05 | 191.26 | 191.26 | optimal | 21.0 | 191.26 | 8 Integer Solution | 60.1 |

## Bottom line

- **Correctness gate: 0 violations** (independent solu cross-check) — clean.
- **discopt: 43/50 proved-optimal**, up from **0/50** at the 06-18 baseline —
  the structural win of this session lands on the curated global set.
- **Remaining gap to BARON** is convergence *speed* (≈13× geomean on jointly
  proved, 4 honest gaps where discopt has the right incumbent but can't close the
  bound in 60 s), plus one data-format artifact (`st_miqp5` binary `.nl`) and two
  no-incumbent time-limits (`casctanks`, `hda`). No correctness debt.
