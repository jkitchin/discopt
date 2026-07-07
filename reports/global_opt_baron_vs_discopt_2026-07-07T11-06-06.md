# Global Optimization Benchmark — discopt vs BARON (GAMS, full license)

- Instances: **61** vendored MINLPLib `.nl` (`python/tests/data/minlplib_nl/`)
- Per-problem time limit: **60 s**; gap tolerance 1e-4; correctness tol abs=1e-6 rel=1e-4 vs MINLPLib `primalbound`
- discopt: isolated subprocess `Model.solve(time_limit=60, gap_tolerance=1e-4)`
- BARON: `gams <name>.gms minlp=baron optcr=0 optca=1e-9 reslim=60` (CMU full license)
- Generated: 2026-07-07T11-06-06

## Verdict vocabulary

| verdict | meaning |
|---|---|
| `ok` | incumbent matches the proven global within tolerance |
| `GAP` | honest feasible/uncertified incumbent **worse** than the global — a convergence gap in the time budget, *not* a correctness bug |
| `VIOLATION` | **the non-negotiable red line**: solver claimed a certified global with the wrong value, returned an incumbent strictly *better* than the proven global, or reported a **dual bound crossing the oracle** (all impossible → bug) |
| `n/a` | no oracle, or no incumbent returned (e.g. parser error) |

## Correctness summary

- Instances with a known optimum (oracle): **58/61**
- **discopt — VIOLATIONS: 0**  ·  ok 50  ·  gap 1  ·  n/a 10
- **BARON   — VIOLATIONS: 0**  ·  ok 0  ·  gap 0  ·  n/a 61

> ✅ **Zero discopt correctness violations.**

## Per-instance results

| instance | known | discopt obj | d | d status | d time | BARON obj | b | b status | b time |
|---|---|---|---|---|---|---|---|---|---|
| 4stufen |   1.1633e+05 |            - | n/a | time_limit | 62.65 |            - | n/a | SKIPPED | 0.00 |
| alan |        2.925 |        2.925 | ok | optimal | 0.17 |            - | n/a | SKIPPED | 0.00 |
| bchoco06 |            - |            - | n/a | time_limit | 62.28 |            - | n/a | SKIPPED | 0.00 |
| bchoco07 |            - |            - | n/a | time_limit | 62.85 |            - | n/a | SKIPPED | 0.00 |
| bchoco08 |            - |            - | n/a | time_limit | 65.60 |            - | n/a | SKIPPED | 0.00 |
| beuster |   1.1633e+05 |            - | n/a | time_limit | 61.05 |            - | n/a | SKIPPED | 0.00 |
| casctanks |       9.1635 |            - | n/a | time_limit | 64.20 |            - | n/a | SKIPPED | 0.00 |
| chance |       29.894 |       29.894 | ok | optimal | 0.39 |            - | n/a | SKIPPED | 0.00 |
| clay0303hfsg |        26669 |        26669 | ok | optimal | 25.25 |            - | n/a | SKIPPED | 0.00 |
| contvar |   8.0915e+05 |   8.0915e+05 | ok | feasible | 60.80 |            - | n/a | SKIPPED | 0.00 |
| cvxnonsep_nsig30 |       130.63 |       130.63 | ok | optimal | 2.76 |            - | n/a | SKIPPED | 0.00 |
| cvxnonsep_psig30 |       78.999 |       78.999 | ok | optimal | 1.38 |            - | n/a | SKIPPED | 0.00 |
| cvxnonsep_psig40r |       86.545 |       86.545 | ok | optimal | 4.15 |            - | n/a | SKIPPED | 0.00 |
| dispatch |       3155.3 |       3155.3 | ok | optimal | 0.48 |            - | n/a | SKIPPED | 0.00 |
| ex1221 |       7.6672 |       7.6672 | ok | optimal | 0.61 |            - | n/a | SKIPPED | 0.00 |
| ex1222 |       1.0765 |       1.0765 | ok | optimal | 0.41 |            - | n/a | SKIPPED | 0.00 |
| ex1224 |     -0.94347 |     -0.94347 | ok | optimal | 2.27 |            - | n/a | SKIPPED | 0.00 |
| ex1225 |           31 |           31 | ok | optimal | 1.23 |            - | n/a | SKIPPED | 0.00 |
| ex1226 |          -17 |          -17 | ok | optimal | 0.54 |            - | n/a | SKIPPED | 0.00 |
| fac2 |   3.3184e+08 |   3.3184e+08 | ok | optimal | 6.74 |            - | n/a | SKIPPED | 0.00 |
| flay02m |       37.947 |       37.947 | ok | optimal | 0.55 |            - | n/a | SKIPPED | 0.00 |
| flay03m |        48.99 |        48.99 | ok | optimal | 7.32 |            - | n/a | SKIPPED | 0.00 |
| gbd |          2.2 |          2.2 | ok | optimal | 0.15 |            - | n/a | SKIPPED | 0.00 |
| gkocis |      -1.9231 |      -1.9231 | ok | optimal | 0.74 |            - | n/a | SKIPPED | 0.00 |
| hda |      -5964.5 |            - | n/a | time_limit | 65.17 |            - | n/a | SKIPPED | 0.00 |
| heatexch_gen1 |    1.549e+05 |            - | n/a | time_limit | 62.16 |            - | n/a | SKIPPED | 0.00 |
| heatexch_gen2 |   6.3584e+05 |   6.7641e+05 | GAP | feasible | 61.21 |            - | n/a | SKIPPED | 0.00 |
| heatexch_gen3 |        64844 |            - | n/a | time_limit | 61.44 |            - | n/a | SKIPPED | 0.00 |
| m3 |         37.8 |         37.8 | ok | optimal | 4.17 |            - | n/a | SKIPPED | 0.00 |
| nvs01 |        12.47 |        12.47 | ok | optimal | 3.09 |            - | n/a | SKIPPED | 0.00 |
| nvs02 |       5.9642 |       5.9642 | ok | optimal | 0.13 |            - | n/a | SKIPPED | 0.00 |
| nvs03 |           16 |           16 | ok | optimal | 0.42 |            - | n/a | SKIPPED | 0.00 |
| nvs04 |         0.72 |         0.72 | ok | optimal | 0.49 |            - | n/a | SKIPPED | 0.00 |
| nvs05 |       5.4709 |       5.4709 | ok | feasible | 60.23 |            - | n/a | SKIPPED | 0.00 |
| nvs06 |       1.7703 |       1.7703 | ok | optimal | 1.60 |            - | n/a | SKIPPED | 0.00 |
| nvs07 |            4 |            4 | ok | optimal | 0.04 |            - | n/a | SKIPPED | 0.00 |
| nvs08 |        23.45 |        23.45 | ok | optimal | 1.05 |            - | n/a | SKIPPED | 0.00 |
| nvs09 |      -43.134 |      -43.134 | ok | feasible | 60.06 |            - | n/a | SKIPPED | 0.00 |
| nvs10 |       -310.8 |       -310.8 | ok | optimal | 0.09 |            - | n/a | SKIPPED | 0.00 |
| nvs11 |         -431 |         -431 | ok | optimal | 0.42 |            - | n/a | SKIPPED | 0.00 |
| nvs12 |       -481.2 |       -481.2 | ok | optimal | 1.60 |            - | n/a | SKIPPED | 0.00 |
| nvs13 |       -585.2 |       -585.2 | ok | optimal | 2.27 |            - | n/a | SKIPPED | 0.00 |
| nvs14 |       -40358 |       -40358 | ok | optimal | 0.29 |            - | n/a | SKIPPED | 0.00 |
| nvs15 |            1 |            1 | ok | optimal | 0.06 |            - | n/a | SKIPPED | 0.00 |
| oaer |      -1.9231 |      -1.9231 | ok | optimal | 1.18 |            - | n/a | SKIPPED | 0.00 |
| st_e13 |            2 |            2 | ok | optimal | 0.47 |            - | n/a | SKIPPED | 0.00 |
| st_e29 |     -0.94347 |     -0.94347 | ok | optimal | 3.03 |            - | n/a | SKIPPED | 0.00 |
| st_e36 |         -246 |         -246 | ok | optimal | 16.26 |            - | n/a | SKIPPED | 0.00 |
| st_e38 |       7197.7 |       7197.7 | ok | optimal | 0.60 |            - | n/a | SKIPPED | 0.00 |
| st_miqp1 |          281 |          281 | ok | optimal | 0.14 |            - | n/a | SKIPPED | 0.00 |
| st_miqp2 |            2 |            2 | ok | optimal | 0.15 |            - | n/a | SKIPPED | 0.00 |
| st_miqp3 |           -6 |           -6 | ok | optimal | 0.14 |            - | n/a | SKIPPED | 0.00 |
| st_miqp4 |        -4574 |        -4574 | ok | optimal | 0.15 |            - | n/a | SKIPPED | 0.00 |
| st_test1 |            0 |  -1.6496e-08 | ok | optimal | 0.15 |            - | n/a | SKIPPED | 0.00 |
| st_testgr3 |       -20.59 |       -20.59 | ok | optimal | 0.22 |            - | n/a | SKIPPED | 0.00 |
| tanksize |       1.2686 |       1.2686 | ok | feasible | 60.09 |            - | n/a | SKIPPED | 0.00 |
| tls2 |          5.3 |            - | n/a | time_limit | 60.46 |            - | n/a | SKIPPED | 0.00 |
| tspn05 |       191.26 |       191.26 | ok | optimal | 19.53 |            - | n/a | SKIPPED | 0.00 |
| tspn08 |       290.57 |       290.57 | ok | feasible | 17.91 |            - | n/a | SKIPPED | 0.00 |
| tspn10 |       225.13 |       225.13 | ok | feasible | 15.63 |            - | n/a | SKIPPED | 0.00 |
| tspn12 |       262.65 |       262.65 | ok | feasible | 14.33 |            - | n/a | SKIPPED | 0.00 |

## discopt convergence gaps (honest, suboptimal in budget)

- **heatexch_gen2**: discopt 6.7641e+05 (`feasible`) vs global 6.3584e+05 — BARON - (also n/a)
