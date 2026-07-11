# Global Optimization Benchmark — discopt vs BARON (GAMS, full license)

- Instances: **50** vendored MINLPLib `.nl` (`python/tests/data/minlplib_nl/`)
- Per-problem time limit: **60 s**; gap tolerance 1e-4; correctness tol abs=1e-6 rel=1e-4 vs MINLPLib `primalbound`
- discopt: isolated subprocess `Model.solve(time_limit=60, gap_tolerance=1e-4)`
- BARON: `gams <name>.gms minlp=baron optcr=0 optca=1e-9 reslim=60` (CMU full license)
- Generated: 2026-07-10T22-02-14

## Verdict vocabulary

| verdict | meaning |
|---|---|
| `ok` | incumbent matches the proven global within tolerance |
| `GAP` | honest feasible/uncertified incumbent **worse** than the global — a convergence gap in the time budget, *not* a correctness bug |
| `VIOLATION` | **the non-negotiable red line**: solver claimed a certified global with the wrong value, returned an incumbent strictly *better* than the proven global, or reported a **dual bound crossing the oracle** (all impossible → bug) |
| `n/a` | no oracle, or no incumbent returned (e.g. parser error) |

## Correctness summary

- Instances with a known optimum (oracle): **50/50**
- **discopt — VIOLATIONS: 0**  ·  ok 48  ·  gap 0  ·  n/a 2
- **BARON   — VIOLATIONS: 0**  ·  ok 0  ·  gap 0  ·  n/a 50

> ✅ **Zero discopt correctness violations.**

## Root-gap instrumentation (TAIL-1c)

Root gap = `|obj − root_bound| / max(1, |obj|)` at the end of root processing (before the first branch), for each solver against its own incumbent. `ratio` = discopt / BARON (the `root_gap_ratio_vs_baron` gate quantity; lower is tighter, gate target ≤ 1.3). Rows with no root bound on either side are excluded from the ratio.

| instance | d root_gap | d root_t | BARON root_gap | b root_t | ratio |
|---|---|---|---|---|---|
| alan |  0.0088757 |  0.13846 |          - |        - |        - |
| casctanks |     10.841 |   30.438 |          - |        - |        - |
| chance |          0 |  0.66397 |          - |        - |        - |
| clay0303hfsg |          1 |   15.341 |          - |        - |        - |
| cvxnonsep_nsig30 |  0.0011386 |   4.4628 |          - |        - |        - |
| cvxnonsep_psig30 |   0.003216 |   1.7261 |          - |        - |        - |
| cvxnonsep_psig40r |  0.0028675 |    4.697 |          - |        - |        - |
| dispatch | 6.8143e-05 |  0.74514 |          - |        - |        - |
| ex1221 | 0.00022683 |   1.1556 |          - |        - |        - |
| ex1222 | 3.4396e-08 |  0.59732 |          - |        - |        - |
| ex1224 |   0.044531 |   4.4573 |          - |        - |        - |
| ex1225 |    0.07802 |   2.5032 |          - |        - |        - |
| ex1226 |    0.01509 |  0.93326 |          - |        - |        - |
| fac2 |    0.22997 |   10.952 |          - |        - |        - |
| flay02m |    0.25464 |  0.83678 |          - |        - |        - |
| flay03m |    0.36754 |   4.2474 |          - |        - |        - |
| gbd | 7.7587e-09 |  0.13006 |          - |        - |        - |
| gkocis |     2.3601 |   1.3702 |          - |        - |        - |
| hda |          - |   60.096 |          - |        - |        - |
| m3 |          1 |   5.9878 |          - |        - |        - |
| nvs01 |    0.64699 |   2.2149 |          - |        - |        - |
| nvs02 |  0.0039848 | 0.0045616 |          - |        - |        - |
| nvs03 |    0.49049 |  0.55564 |          - |        - |        - |
| nvs04 |     0.8725 |   0.7459 |          - |        - |        - |
| nvs05 |     0.8768 |   2.5203 |          - |        - |        - |
| nvs06 |    0.26566 |   1.0892 |          - |        - |        - |
| nvs07 |     0.2497 | 0.0018573 |          - |        - |        - |
| nvs08 |    0.25376 |   1.1941 |          - |        - |        - |
| nvs09 |    0.69009 |   3.5808 |          - |        - |        - |
| nvs10 |    0.27735 | 0.0055385 |          - |        - |        - |
| nvs11 |      5.453 | 0.027218 |          - |        - |        - |
| nvs12 |     6.3867 | 0.086548 |          - |        - |        - |
| nvs13 |  0.0088346 |   1.2782 |          - |        - |        - |
| nvs14 |  0.0058888 | 0.0048653 |          - |        - |        - |
| nvs15 |     3.9975 | 0.001747 |          - |        - |        - |
| oaer |     4.7199 |   1.3454 |          - |        - |        - |
| st_e13 |    0.02381 |  0.55974 |          - |        - |        - |
| st_e29 |   0.044531 |   4.4019 |          - |        - |        - |
| st_e36 |     0.2378 |   4.7509 |          - |        - |        - |
| st_e38 |   0.040846 |  0.94546 |          - |        - |        - |
| st_miqp1 |    0.14567 |  0.13385 |          - |        - |        - |
| st_miqp2 |     3.8151 |  0.13244 |          - |        - |        - |
| st_miqp3 |          0 |  0.13292 |          - |        - |        - |
| st_miqp4 | 0.00054657 |  0.13621 |          - |        - |        - |
| st_miqp5 |          0 |  0.14995 |          - |        - |        - |
| st_test1 |     32.006 |  0.13194 |          - |        - |        - |
| st_testgr3 |  0.0063126 |  0.14998 |          - |        - |        - |
| tanksize |    0.33211 |   5.7983 |          - |        - |        - |
| tls2 |          - |   9.6178 |          - |        - |        - |
| tspn05 |    0.12269 |   7.6589 |          - |        - |        - |

- discopt root_gap populated: **48/50**  ·  BARON root_gap populated: **0/50**
- Mean root_gap ratio (discopt/BARON) over **0** co-populated instances: **-** (gate `root_gap_ratio_vs_baron` target ≤ 1.3)

## Per-instance results

| instance | known | discopt obj | d | d status | d time | BARON obj | b | b status | b time |
|---|---|---|---|---|---|---|---|---|---|
| alan |        2.925 |        2.925 | ok | optimal | 0.23 |            - | n/a | SKIPPED | 0.00 |
| casctanks |       9.1635 |       9.1635 | ok | feasible | 60.50 |            - | n/a | SKIPPED | 0.00 |
| chance |       29.894 |       29.894 | ok | optimal | 0.68 |            - | n/a | SKIPPED | 0.00 |
| clay0303hfsg |        26669 |        26669 | ok | feasible | 60.69 |            - | n/a | SKIPPED | 0.00 |
| cvxnonsep_nsig30 |       130.63 |       130.63 | ok | optimal | 8.58 |            - | n/a | SKIPPED | 0.00 |
| cvxnonsep_psig30 |       78.999 |       78.999 | ok | optimal | 2.77 |            - | n/a | SKIPPED | 0.00 |
| cvxnonsep_psig40r |       86.545 |       86.545 | ok | optimal | 8.44 |            - | n/a | SKIPPED | 0.00 |
| dispatch |       3155.3 |       3155.3 | ok | optimal | 0.77 |            - | n/a | SKIPPED | 0.00 |
| ex1221 |       7.6672 |       7.6672 | ok | optimal | 1.19 |            - | n/a | SKIPPED | 0.00 |
| ex1222 |       1.0765 |       1.0765 | ok | optimal | 0.62 |            - | n/a | SKIPPED | 0.00 |
| ex1224 |     -0.94347 |     -0.94347 | ok | optimal | 5.20 |            - | n/a | SKIPPED | 0.00 |
| ex1225 |           31 |           31 | ok | optimal | 2.55 |            - | n/a | SKIPPED | 0.00 |
| ex1226 |          -17 |          -17 | ok | optimal | 0.97 |            - | n/a | SKIPPED | 0.00 |
| fac2 |   3.3184e+08 |   3.3184e+08 | ok | optimal | 15.52 |            - | n/a | SKIPPED | 0.00 |
| flay02m |       37.947 |       37.947 | ok | optimal | 0.95 |            - | n/a | SKIPPED | 0.00 |
| flay03m |        48.99 |        48.99 | ok | optimal | 7.28 |            - | n/a | SKIPPED | 0.00 |
| gbd |          2.2 |          2.2 | ok | optimal | 0.16 |            - | n/a | SKIPPED | 0.00 |
| gkocis |      -1.9231 |      -1.9231 | ok | optimal | 1.45 |            - | n/a | SKIPPED | 0.00 |
| hda |      -5964.5 |            - | n/a | time_limit | 63.16 |            - | n/a | SKIPPED | 0.00 |
| m3 |         37.8 |         37.8 | ok | optimal | 11.04 |            - | n/a | SKIPPED | 0.00 |
| nvs01 |        12.47 |        12.47 | ok | optimal | 2.74 |            - | n/a | SKIPPED | 0.00 |
| nvs02 |       5.9642 |       5.9642 | ok | optimal | 0.35 |            - | n/a | SKIPPED | 0.00 |
| nvs03 |           16 |           16 | ok | optimal | 0.60 |            - | n/a | SKIPPED | 0.00 |
| nvs04 |         0.72 |         0.72 | ok | optimal | 0.77 |            - | n/a | SKIPPED | 0.00 |
| nvs05 |       5.4709 |       5.4709 | ok | feasible | 60.17 |            - | n/a | SKIPPED | 0.00 |
| nvs06 |       1.7703 |       1.7703 | ok | optimal | 1.14 |            - | n/a | SKIPPED | 0.00 |
| nvs07 |            4 |            4 | ok | optimal | 0.05 |            - | n/a | SKIPPED | 0.00 |
| nvs08 |        23.45 |        23.45 | ok | optimal | 1.54 |            - | n/a | SKIPPED | 0.00 |
| nvs09 |      -43.134 |      -43.134 | ok | feasible | 61.39 |            - | n/a | SKIPPED | 0.00 |
| nvs10 |       -310.8 |       -310.8 | ok | optimal | 0.09 |            - | n/a | SKIPPED | 0.00 |
| nvs11 |         -431 |         -431 | ok | optimal | 0.47 |            - | n/a | SKIPPED | 0.00 |
| nvs12 |       -481.2 |       -481.2 | ok | optimal | 1.50 |            - | n/a | SKIPPED | 0.00 |
| nvs13 |       -585.2 |       -585.2 | ok | optimal | 2.52 |            - | n/a | SKIPPED | 0.00 |
| nvs14 |       -40358 |       -40358 | ok | optimal | 0.33 |            - | n/a | SKIPPED | 0.00 |
| nvs15 |            1 |            1 | ok | optimal | 0.06 |            - | n/a | SKIPPED | 0.00 |
| oaer |      -1.9231 |      -1.9231 | ok | optimal | 1.39 |            - | n/a | SKIPPED | 0.00 |
| st_e13 |            2 |            2 | ok | optimal | 0.59 |            - | n/a | SKIPPED | 0.00 |
| st_e29 |     -0.94347 |     -0.94347 | ok | optimal | 5.14 |            - | n/a | SKIPPED | 0.00 |
| st_e36 |         -246 |         -246 | ok | optimal | 16.27 |            - | n/a | SKIPPED | 0.00 |
| st_e38 |       7197.7 |       7197.7 | ok | optimal | 0.98 |            - | n/a | SKIPPED | 0.00 |
| st_miqp1 |          281 |          281 | ok | optimal | 0.17 |            - | n/a | SKIPPED | 0.00 |
| st_miqp2 |            2 |            2 | ok | optimal | 0.17 |            - | n/a | SKIPPED | 0.00 |
| st_miqp3 |           -6 |           -6 | ok | optimal | 0.14 |            - | n/a | SKIPPED | 0.00 |
| st_miqp4 |        -4574 |        -4574 | ok | optimal | 0.17 |            - | n/a | SKIPPED | 0.00 |
| st_miqp5 |      -333.89 |      -333.89 | ok | optimal | 0.17 |            - | n/a | SKIPPED | 0.00 |
| st_test1 |            0 |  -1.6496e-08 | ok | optimal | 0.18 |            - | n/a | SKIPPED | 0.00 |
| st_testgr3 |       -20.59 |       -20.59 | ok | optimal | 0.82 |            - | n/a | SKIPPED | 0.00 |
| tanksize |       1.2686 |       1.2686 | ok | feasible | 60.21 |            - | n/a | SKIPPED | 0.00 |
| tls2 |          5.3 |            - | n/a | time_limit | 60.67 |            - | n/a | SKIPPED | 0.00 |
| tspn05 |       191.26 |       191.26 | ok | optimal | 27.35 |            - | n/a | SKIPPED | 0.00 |
