# #1004 — captured probe output

Run 2026-08-13. The `.log` files these tables come from are gitignored by repo
convention (`.gitignore`: `scratchpad/**/*.log`), so the summaries are captured
here and the per-configuration detail lives in the committed `*_results.json`.

## E1 detection panel (all 12 gdplib_small models)

```
UNBIASED POOL (random + neighbour configs)
model                 cfgs  feas         zero        relax           random
--------------------------------------------------------------------------------------
jobshop                  2     0      0/0 n/a      0/0 n/a          0/0 n/a
ex1_linan_2023          63     2     2/2 100%     2/2 100%       16/16 100%
positioning             63    63   63/63 100%   63/63 100%     504/504 100%
small_batch             58     0      0/0 n/a      0/0 n/a          0/0 n/a
cstr                    61     0      0/0 n/a      0/0 n/a          0/0 n/a
spectralog              52    52   52/52 100%   52/52 100%     416/416 100%
methanol                 4     4     4/4 100%     4/4 100%        31/32 97%
batch_processing        69    64   64/64 100%   64/64 100%         0/512 0%
syngas                  60     0      0/0 n/a      0/0 n/a          0/0 n/a
water_network           26     7      6/7 86%       0/7 0%         6/56 11%
gdp_col                 58     0      0/0 n/a      0/0 n/a          0/0 n/a
modprodnet               1     1     1/1 100%     1/1 100%         8/8 100%
--------------------------------------------------------------------------------------
TOTAL                  517   193 192/193   99%   186/193   96%   981/1544   64%
  zero-start detection on configurations proven feasible WITHOUT it: 188/189 99%  |  found only by the zero start: 4  |  missed by the zero start: 1

======================================================================================
DIVE-DERIVED POOL (biased toward the zero start)
model                 cfgs  feas         zero        relax           random
--------------------------------------------------------------------------------------
jobshop                  6     6     6/6 100%     6/6 100%       48/48 100%
ex1_linan_2023           0     0      0/0 n/a      0/0 n/a          0/0 n/a
positioning             63    63   63/63 100%   63/63 100%     504/504 100%
small_batch             10    10   10/10 100%   10/10 100%       80/80 100%
cstr                    12    12   12/12 100%    10/12 83%        29/96 30%
spectralog              40    40   40/40 100%   40/40 100%     320/320 100%
methanol                12    12   12/12 100%   12/12 100%        93/96 97%
batch_processing         2     2     2/2 100%     2/2 100%          0/16 0%
syngas                   0     0      0/0 n/a      0/0 n/a          0/0 n/a
water_network            1     1     1/1 100%       0/1 0%          3/8 38%
gdp_col                  5     5      3/5 60%     5/5 100%          1/40 2%
modprodnet               1     1     1/1 100%     1/1 100%         8/8 100%
--------------------------------------------------------------------------------------
TOTAL                  152   152 150/152   99%   149/152   98%   1086/1216   89%
  zero-start detection on configurations proven feasible WITHOUT it: 150/152 99%  |  found only by the zero start: 0  |  missed by the zero start: 2
======================================================================================
TOTAL executed feasibility tests: 6690
wrote scratchpad/issue1004/E1_results.json
```

## E1 deep pass (dive disabled, 600-configuration request)

```
UNBIASED POOL (random + neighbour configs)
model                 cfgs  feas         zero        relax           random
--------------------------------------------------------------------------------------
jobshop                  8     6     6/6 100%     6/6 100%       24/24 100%
small_batch            460     2     2/2 100%     2/2 100%         8/8 100%
cstr                   447     1     1/1 100%     1/1 100%          1/4 25%
--------------------------------------------------------------------------------------
TOTAL                  915     9 9/9  100%   9/9  100%   33/36   92%
  zero-start detection on configurations proven feasible WITHOUT it: 9/9 100%  |  found only by the zero start: 0  |  missed by the zero start: 0

======================================================================================
DIVE-DERIVED POOL (biased toward the zero start)
model                 cfgs  feas         zero        relax           random
--------------------------------------------------------------------------------------
jobshop                  0     0      0/0 n/a      0/0 n/a          0/0 n/a
small_batch              0     0      0/0 n/a      0/0 n/a          0/0 n/a
cstr                     0     0      0/0 n/a      0/0 n/a          0/0 n/a
--------------------------------------------------------------------------------------
TOTAL                    0     0 0/0   n/a   0/0   n/a   0/0   n/a
  zero-start detection on configurations proven feasible WITHOUT it: 0/0 n/a  |  found only by the zero start: 0  |  missed by the zero start: 0
======================================================================================
TOTAL executed feasibility tests: 5490
wrote scratchpad/issue1004/E1_deep_results.json
```

## E2 restart cost — different starts

```
load before run: (0.55810546875, 1.142578125, 1.3154296875)
[small_batch] 3 config(s), 45 solves | start 1: 21.3 ± 2.2 ms (n=9) | starts 2..5: 18.1 ± 2.3 ms (n=36) | ratio 0.848
[cstr] 3 config(s), 45 solves | start 1: 47.2 ± 16.0 ms (n=9) | starts 2..5: 156.1 ± 83.2 ms (n=36) | ratio 3.308
[spectralog] 3 config(s), 45 solves | start 1: 29.4 ± 2.0 ms (n=9) | starts 2..5: 52.0 ± 3.4 ms (n=36) | ratio 1.767
[batch_processing] 2 config(s), 30 solves | start 1: 271.2 ± 15.6 ms (n=6) | starts 2..5: 4.9 ± 0.3 ms (n=24) | ratio 0.018
[syngas] no feasible configuration found in 45.0s — skipped
[gdp_col] 3 config(s), 45 solves | start 1: 1588.1 ± 913.4 ms (n=9) | starts 2..5: 1789.2 ± 322.8 ms (n=36) | ratio 1.127
load after run: (1.2705078125, 1.20849609375, 1.28515625)
====================================================================================
model                 solves      start1 (ms)    starts2..k (ms)    ratio
------------------------------------------------------------------------------------
small_batch               45      21.3 ± 2.2          18.1 ± 2.3      0.848
cstr                      45      47.2 ± 16.0        156.1 ± 83.2     3.308
spectralog                45      29.4 ± 2.0          52.0 ± 3.4      1.767
batch_processing          30     271.2 ± 15.6          4.9 ± 0.3      0.018
gdp_col                   45    1588.1 ± 913.4      1789.2 ± 322.8    1.127
------------------------------------------------------------------------------------
TOTAL executed sub-NLP solves: 210
====================================================================================
wrote scratchpad/issue1004/E2_results.json
```

## E2 restart cost — same-start control (the decisive arm)

```
load before run: (1.2705078125, 1.20849609375, 1.28515625)
[small_batch] 3 config(s), 45 solves | start 1: 20.7 ± 1.3 ms (n=9) | starts 2..5: 20.0 ± 1.0 ms (n=36) | ratio 0.967
[cstr] 3 config(s), 45 solves | start 1: 41.1 ± 7.3 ms (n=9) | starts 2..5: 39.3 ± 4.1 ms (n=36) | ratio 0.954
[spectralog] 3 config(s), 45 solves | start 1: 29.8 ± 3.1 ms (n=9) | starts 2..5: 30.6 ± 2.3 ms (n=36) | ratio 1.026
[batch_processing] 2 config(s), 30 solves | start 1: 260.2 ± 13.0 ms (n=6) | starts 2..5: 262.8 ± 13.7 ms (n=24) | ratio 1.010
[syngas] no feasible configuration found in 45.0s — skipped
[gdp_col] 3 config(s), 45 solves | start 1: 1354.1 ± 404.0 ms (n=9) | starts 2..5: 1360.6 ± 404.1 ms (n=36) | ratio 1.005
load after run: (1.1455078125, 1.1767578125, 1.25146484375)
====================================================================================
model                 solves      start1 (ms)    starts2..k (ms)    ratio
------------------------------------------------------------------------------------
small_batch               45      20.7 ± 1.3          20.0 ± 1.0      0.967
cstr                      45      41.1 ± 7.3          39.3 ± 4.1      0.954
spectralog                45      29.8 ± 3.1          30.6 ± 2.3      1.026
batch_processing          30     260.2 ± 13.0        262.8 ± 13.7     1.010
gdp_col                   45    1354.1 ± 404.0      1360.6 ± 404.1    1.005
------------------------------------------------------------------------------------
TOTAL executed sub-NLP solves: 210
====================================================================================
wrote scratchpad/issue1004/E2_samestart_results.json
```
