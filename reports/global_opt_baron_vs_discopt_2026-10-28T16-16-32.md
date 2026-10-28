# Global Optimization Benchmark — discopt vs BARON (GAMS, full license)

- Instances: **50** vendored MINLPLib `.nl` (`python/tests/data/minlplib_nl/`)
- Per-problem time limit: **60 s**; gap tolerance 1e-4; correctness tol abs=1e-6 rel=1e-4 vs MINLPLib `primalbound`
- discopt: isolated subprocess `Model.solve(time_limit=60, gap_tolerance=1e-4)`
- BARON: `gams <name>.gms minlp=baron optcr=0 optca=1e-9 reslim=60` (CMU full license)
- Generated: 2026-10-28T16-16-32
- geomean wall/BARON: **11.1x** (n=28)
- geomean solve/BARON (import floor excluded): **7.3x** (n=28)

## Verdict vocabulary

| verdict | meaning |
|---|---|
| `ok` | incumbent matches the proven global within tolerance |
| `GAP` | honest feasible/uncertified incumbent **worse** than the global — a convergence gap in the time budget, *not* a correctness bug |
| `VIOLATION` | **the non-negotiable red line**: solver claimed a certified global with the wrong value, returned an incumbent strictly *better* than the proven global, or reported a **dual bound crossing the oracle** (all impossible → bug) |
| `n/a` | no oracle, or no incumbent returned (e.g. parser error) |

## Correctness summary

- Instances with a known optimum (oracle): **50/50**
- **discopt — VIOLATIONS: 0**  ·  ok 48  ·  gap 1  ·  n/a 1
- **BARON   — VIOLATIONS: 1**  ·  ok 49  ·  gap 0  ·  n/a 0

### Solved vs certified-claim (score capability by *solved*)

- **Solved correctly** (objective matches oracle — the fair capability metric): discopt **48**  ·  BARON **49**
- Certified-global *claim* (status only): discopt 48  ·  BARON 45
> BARON-via-GAMS reports `2 Locally Optimal` / `8 Integer Solution` on many instances it has solved to the global optimum, so its *claim* count undercounts it (~2x). **Rank by *solved* (`ok`), not by the claim count.**

> ✅ **Zero discopt correctness violations.**

## Root-gap instrumentation (TAIL-1c)

Root gap = `|obj − root_bound| / max(1, |obj|)` at the end of root processing (before the first branch), for each solver against its own incumbent. `ratio` = discopt / BARON (the `root_gap_ratio_vs_baron` gate quantity; lower is tighter, gate target ≤ 1.3). Rows with no root bound on either side are excluded from the ratio.

| instance | d root_gap | d root_t | BARON root_gap | b root_t | ratio |
|---|---|---|---|---|---|
| alan |  0.0088757 | 0.030769 |  0.0088752 |     0.05 |   1.0001 |
| casctanks |     10.794 |   18.615 |          1 |     1.33 |   10.794 |
| chance |  4.133e-10 | 0.018995 |          - |        - |        - |
| clay0303hfsg |          - |        - |    0.86651 |      0.1 |        - |
| cvxnonsep_nsig30 |  0.0011387 |   35.685 |  0.0011383 |     0.07 |   1.0003 |
| cvxnonsep_psig30 |   0.003216 |   34.758 |  0.0032165 |     0.05 |  0.99983 |
| cvxnonsep_psig40r |          - |        - |  0.0028679 |     0.05 |        - |
| dispatch | 1.6946e-07 | 0.00041058 |          - |        - |        - |
| ex1221 | 1.8187e-14 |   0.1075 |          - |        - |        - |
| ex1222 | 3.3207e-14 | 0.048081 |          - |        - |        - |
| ex1224 |   0.025757 |  0.29969 |      3e-05 |     0.04 |   858.57 |
| ex1225 |   0.076951 |  0.90267 |          0 |     0.03 |        - |
| ex1226 |    0.23529 |   1.2795 |   0.067218 |     0.04 |   3.5005 |
| fac2 |          - |        - | 1.5013e-06 |     0.04 |        - |
| flay02m |          - |        - |   0.044765 |     0.02 |        - |
| flay03m |          - |        - |   0.058937 |     0.03 |        - |
| gbd |          - |        - |          0 |     0.02 |        - |
| gkocis |     2.2759 |   0.2783 |          0 |     0.03 |        - |
| hda |          - |   59.206 |          - |        - |        - |
| m3 |          - |        - |          0 |     0.03 |        - |
| nvs01 |    0.60047 |  0.74714 |   0.063843 |     0.04 |   9.4055 |
| nvs02 |  0.0039848 | 0.0024393 | 3.3533e-06 |     0.03 |   1188.3 |
| nvs03 |          - |        - |          0 |     0.02 |        - |
| nvs04 |       0.72 | 0.070813 |          0 |     0.03 |        - |
| nvs05 |    0.87493 |   2.8855 |    0.21709 |     0.05 |   4.0302 |
| nvs06 |    0.26566 |  0.19128 | 5.6488e-06 |     0.03 |    47029 |
| nvs07 |     0.2497 | 0.0010242 |          0 |     0.03 |        - |
| nvs08 |    0.30654 |  0.35432 |          0 |     0.04 |        - |
| nvs09 |    0.24441 |  0.52222 |          0 |     0.02 |        - |
| nvs10 |          - |        - |  0.0070785 |     0.02 |        - |
| nvs11 |          - |        - |  0.0039443 |     0.02 |        - |
| nvs12 |          - |        - |  0.0039485 |     0.02 |        - |
| nvs13 |  0.0065115 | 0.0060938 |   0.008715 |     0.04 |  0.74717 |
| nvs14 |  0.0058888 | 0.002398 |   1.12e-06 |     0.03 |     5258 |
| nvs15 |          - |        - |          0 |     0.02 |        - |
| oaer |    0.36853 |  0.21709 |          0 |     0.03 |        - |
| st_e13 | 1.2609e-09 | 6.875e-06 |          0 |     0.03 |        - |
| st_e29 |   0.025757 |   0.2922 |      3e-05 |     0.04 |   858.57 |
| st_e36 |     0.2378 |    1.409 |     0.2373 |     0.06 |   1.0021 |
| st_e38 |  0.0059467 | 0.00028833 |   0.097735 |     0.03 | 0.060846 |
| st_miqp1 |          - |        - |          0 |     0.02 |        - |
| st_miqp2 |          - |        - |        3.5 |     0.02 |        - |
| st_miqp3 |          - |        - |          - |        - |        - |
| st_miqp4 |          - |        - | 4.3725e-06 |     0.02 |        - |
| st_miqp5 |          - |        - |  2.995e-07 |     0.02 |        - |
| st_test1 |          - |        - |          0 |     0.02 |        - |
| st_testgr3 |          - |        - |  0.0063137 |     0.02 |        - |
| tanksize |    0.33751 | 0.00073892 |    0.24695 |      0.1 |   1.3667 |
| tls2 |          - |        - |          0 |     0.03 |        - |
| tspn05 |    0.12269 |  0.33185 | 0.00084285 |     0.06 |   145.57 |

- discopt root_gap populated: **29/50**  ·  BARON root_gap populated: **44/50**
- Mean root_gap ratio (discopt/BARON) over **17** co-populated instances: **3257.2** (gate `root_gap_ratio_vs_baron` target ≤ 1.3)

## Per-instance results

| instance | known | discopt obj | d | d status | d time | BARON obj | b | b status | b time |
|---|---|---|---|---|---|---|---|---|---|
| alan |        2.925 |        2.925 | ok | optimal | 0.69 |        2.925 | ok | 1 Optimal | 0.10 |
| casctanks |       9.1635 |       9.2076 | GAP | feasible | 60.53 |       9.1635 | ok | 8 Integer Solution | 61.10 |
| chance |       29.894 |       29.894 | ok | optimal | 0.68 |       29.894 | ok | 2 Locally Optimal | 0.01 |
| clay0303hfsg |        26669 |        26669 | ok | optimal | 2.03 |        26669 | ok | 1 Optimal | 1.45 |
| cvxnonsep_nsig30 |       130.63 |       130.63 | ok | optimal | 36.99 |       130.63 | ok | 1 Optimal | 1.84 |
| cvxnonsep_psig30 |       78.999 |       78.999 | ok | optimal | 35.59 |       78.999 | ok | 1 Optimal | 0.78 |
| cvxnonsep_psig40r |       86.545 |       86.545 | ok | optimal | 1.09 |       86.545 | ok | 1 Optimal | 0.38 |
| dispatch |       3155.3 |       3155.3 | ok | optimal | 0.62 |       3155.3 | ok | 2 Locally Optimal | 0.01 |
| ex1221 |       7.6672 |       7.6672 | ok | optimal | 0.66 |       7.6672 | ok | 1 Optimal | 0.02 |
| ex1222 |       1.0765 |       1.0765 | ok | optimal | 0.60 |       1.0765 | ok | 1 Optimal | 0.01 |
| ex1224 |     -0.94347 |     -0.94347 | ok | optimal | 0.90 |      -0.9435 | ok | 1 Optimal | 0.06 |
| ex1225 |           31 |           31 | ok | optimal | 1.53 |           31 | ok | 1 Optimal | 0.04 |
| ex1226 |          -17 |          -17 | ok | optimal | 1.87 |          -17 | ok | 1 Optimal | 0.05 |
| fac2 |   3.3184e+08 |   3.3184e+08 | ok | optimal | 0.82 |   3.3184e+08 | ok | 1 Optimal | 0.04 |
| flay02m |       37.947 |       37.947 | ok | optimal | 0.66 |       37.947 | ok | 1 Optimal | 0.04 |
| flay03m |        48.99 |        48.99 | ok | optimal | 0.88 |        48.99 | ok | 1 Optimal | 0.30 |
| gbd |          2.2 |          2.2 | ok | optimal | 0.59 |          2.2 | ok | 1 Optimal | 0.02 |
| gkocis |      -1.9231 |      -1.9231 | ok | optimal | 0.89 |      -1.9231 | ok | 1 Optimal | 0.04 |
| hda |      -5964.5 |            - | n/a | time_limit | 60.84 |            - | VIOLATION | 19 Infeasible - No Solution | 0.22 |
| m3 |         37.8 |         37.8 | ok | optimal | 0.73 |         37.8 | ok | 1 Optimal | 0.03 |
| nvs01 |        12.47 |        12.47 | ok | optimal | 1.40 |        12.47 | ok | 1 Optimal | 0.05 |
| nvs02 |       5.9642 |       5.9642 | ok | optimal | 0.68 |       5.9642 | ok | 1 Optimal | 0.03 |
| nvs03 |           16 |           16 | ok | optimal | 0.64 |           16 | ok | 1 Optimal | 0.02 |
| nvs04 |         0.72 |         0.72 | ok | optimal | 0.63 |         0.72 | ok | 1 Optimal | 0.03 |
| nvs05 |       5.4709 |       5.4709 | ok | optimal | 18.17 |       5.4709 | ok | 1 Optimal | 0.53 |
| nvs06 |       1.7703 |       1.7703 | ok | optimal | 0.85 |       1.7703 | ok | 1 Optimal | 0.04 |
| nvs07 |            4 |            4 | ok | optimal | 0.59 |            4 | ok | 1 Optimal | 0.03 |
| nvs08 |        23.45 |        23.45 | ok | optimal | 0.96 |        23.45 | ok | 1 Optimal | 0.05 |
| nvs09 |      -43.134 |      -43.134 | ok | optimal | 3.91 |      -43.134 | ok | 1 Optimal | 0.02 |
| nvs10 |       -310.8 |       -310.8 | ok | optimal | 0.70 |       -310.8 | ok | 1 Optimal | 0.03 |
| nvs11 |         -431 |         -431 | ok | optimal | 0.64 |         -431 | ok | 1 Optimal | 0.04 |
| nvs12 |       -481.2 |       -481.2 | ok | optimal | 0.72 |       -481.2 | ok | 1 Optimal | 0.05 |
| nvs13 |       -585.2 |       -585.2 | ok | optimal | 0.99 |       -585.2 | ok | 1 Optimal | 0.15 |
| nvs14 |       -40358 |       -40358 | ok | optimal | 0.65 |       -40358 | ok | 1 Optimal | 0.04 |
| nvs15 |            1 |            1 | ok | optimal | 0.64 |            1 | ok | 1 Optimal | 0.02 |
| oaer |      -1.9231 |      -1.9231 | ok | optimal | 0.79 |      -1.9231 | ok | 1 Optimal | 0.04 |
| st_e13 |            2 |            2 | ok | optimal | 0.57 |            2 | ok | 1 Optimal | 0.03 |
| st_e29 |     -0.94347 |     -0.94347 | ok | optimal | 0.88 |      -0.9435 | ok | 1 Optimal | 0.05 |
| st_e36 |         -246 |         -246 | ok | optimal | 4.80 |         -246 | ok | 1 Optimal | 0.08 |
| st_e38 |       7197.7 |       7197.7 | ok | optimal | 0.72 |       7197.7 | ok | 1 Optimal | 0.03 |
| st_miqp1 |          281 |          281 | ok | optimal | 0.60 |          281 | ok | 1 Optimal | 0.02 |
| st_miqp2 |            2 |            2 | ok | optimal | 0.61 |            2 | ok | 1 Optimal | 0.02 |
| st_miqp3 |           -6 |           -6 | ok | optimal | 0.60 |           -6 | ok | 1 Optimal | 0.01 |
| st_miqp4 |        -4574 |        -4574 | ok | optimal | 0.60 |        -4574 | ok | 1 Optimal | 0.04 |
| st_miqp5 |      -333.89 |      -333.89 | ok | optimal | 0.61 |      -333.89 | ok | 1 Optimal | 0.02 |
| st_test1 |            0 |            0 | ok | optimal | 0.64 |            0 | ok | 1 Optimal | 0.02 |
| st_testgr3 |       -20.59 |       -20.59 | ok | optimal | 0.68 |       -20.59 | ok | 1 Optimal | 0.03 |
| tanksize |       1.2686 |       1.2686 | ok | optimal | 1.48 |       1.2686 | ok | 1 Optimal | 2.63 |
| tls2 |          5.3 |          5.3 | ok | optimal | 1.01 |          5.3 | ok | 1 Optimal | 0.03 |
| tspn05 |       191.26 |       191.26 | ok | optimal | 1.95 |       191.26 | ok | 8 Integer Solution | 60.06 |

## ⚠️ false infeasibility claims (published global exists)

- **hda**: BARON returned `19 Infeasible - No Solution` in 0.22 s, but the published global is -5964.5 — scored VIOLATION, not a miss

## discopt convergence gaps (honest, suboptimal in budget)

- **casctanks**: discopt 9.2076 (`feasible`) vs global 9.1635 — BARON 9.1635 (global)
