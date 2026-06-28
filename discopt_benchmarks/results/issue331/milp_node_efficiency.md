# MILP node-efficiency bench (issue #331, Step 1)

SCIP version: `10.0` · discopt config: `prod` defaults

## Reproduce — discopt vs SCIP (production config)

| instance | discopt nodes | discopt wall | SCIP nodes | SCIP wall | node ratio | obj match |
|---|---|---|---|---|---|---|
| mdk30x5 | 47 | 0.009s | 1 | 0.095s | 47.0x | True |
| mdk40x5 | 63 | 0.004s | 1 | 0.012s | 63.0x | True |
| mdk50x8 | 61 | 0.009s | 1 | 0.018s | 61.0x | True |
| mdk60x8 | 213 | 0.027s | 27 | 0.237s | 7.9x | True |
| mdk70x10 | 57 | 0.028s | 1 | 0.203s | 57.0x | True |
| mdk90x12 | 197 | 0.051s | 66 | 0.273s | 3.0x | True |
| mdk120x15 | 4713 | 0.509s | 2330 | 3.564s | 2.0x | True |
| mdk150x20 | 1797 | 0.661s | 579 | 1.468s | 3.1x | True |
| mdk200x25 | 17935 | 4.276s | 12688 | 16.610s | 1.4x | True |

`*` = hit per-solve time/node cap (not proven optimal); `obj match = n/a` when a solver did not prove optimality.


## Attribute — root bound & integrality gap closed

| instance | z_LP | z_opt | discopt root bound | discopt gap closed | SCIP gap closed |
|---|---|---|---|---|---|
| mdk30x5 | -921.41 | -902.00 | -912.57 | 45.5% | 83.8% |
| mdk40x5 | -1633.46 | -1620.00 | -1628.99 | 33.2% | 100.0% |
| mdk50x8 | -1769.45 | -1759.14 | -1764.04 | 52.5% | 101.4% |
| mdk60x8 | -2300.14 | -2276.00 | -2294.19 | 24.6% | 39.0% |
| mdk70x10 | -2349.74 | -2336.00 | -2346.38 | 24.5% | 64.7% |
| mdk90x12 | -3350.98 | -3329.00 | -3347.92 | 14.0% | 32.1% |
| mdk120x15 | -4638.87 | -4614.39 | -4636.58 | 9.3% | 19.9% |
| mdk150x20 | -5655.69 | -5635.28 | -5653.02 | 13.1% | 24.5% |
| mdk200x25 | -7428.07 | -7397.74 | -7426.55 | 5.0% | 10.9% |

## Ablation — node count per lever (one lever on top of baseline)

| instance | baseline | presolve | root_cuts | cut_rounds | node_cuts | strong_branch | heuristics | reduced_cost_fixing | prod | full |
|---|---|---|---|---|---|---|---|---|---|---|
| mdk30x5 | 1319 | 1319 | 1415 | 1565 | 1451 | 1461 | 147 | 89 | 47 | 33 |
| mdk40x5 | 1547 | 1547 | 1287 | 769 | 1027 | 1671 | 191 | 31 | 63 | 31 |
| mdk50x8 | 131 | 131 | 641 | 35 | 255 | 259 | 41 | 41 | 61 | 35 |
| mdk60x8 | 1037 | 1037 | 1221 | 1309 | 1101 | 1997 | 605 | 257 | 213 | 135 |
| mdk70x10 | 4607 | 4607 | 4607 | 3457 | 4863 | 4869 | 195 | 91 | 57 | 47 |
| mdk90x12 | 2311 | 2311 | 2551 | 2757 | 2479 | 2673 | 673 | 315 | 197 | 313 |
| mdk120x15 | 20735 | 20735 | 20585 | 21503 | 19841 | 16895 | 10299 | 6331 | 4713 | 4879 |
| mdk150x20 | 10751 | 10751 | 10623 | 11007 | 10239 | 10879 | 3125 | 1973 | 1797 | 1931 |
| mdk200x25 | 58885 | 58885 | 57855 | 57229 | 57855 | 57089 | 29451 | 20235 | 17935 | 19361 |

`*` = config hit node/time limit (not proven optimal). For capped rows the node count is *nodes explored within the budget*, not nodes-to-proof, so it conflates node-efficiency with per-node cost (stronger cuts make each node more expensive); read attribution primarily from the uncapped rows and cross-check wall time.

## Ablation — node reduction attributable to each lever (vs baseline)

| instance | presolve | root_cuts | cut_rounds | node_cuts | strong_branch | heuristics | reduced_cost_fixing | prod | full |
|---|---|---|---|---|---|---|---|---|---|
| mdk30x5 | 0% | -7% | -19% | -10% | -11% | 89% | 93% | 96% | 97% |
| mdk40x5 | 0% | 17% | 50% | 34% | -8% | 88% | 98% | 96% | 98% |
| mdk50x8 | 0% | -389% | 73% | -95% | -98% | 69% | 69% | 53% | 73% |
| mdk60x8 | 0% | -18% | -26% | -6% | -93% | 42% | 75% | 79% | 87% |
| mdk70x10 | 0% | 0% | 25% | -6% | -6% | 96% | 98% | 99% | 99% |
| mdk90x12 | 0% | -10% | -19% | -7% | -16% | 71% | 86% | 91% | 86% |
| mdk120x15 | 0% | 1% | -4% | 4% | 19% | 50% | 69% | 77% | 76% |
| mdk150x20 | 0% | 1% | -2% | 5% | -1% | 71% | 82% | 83% | 82% |
| mdk200x25 | 0% | 2% | 3% | 2% | 3% | 50% | 66% | 70% | 67% |
