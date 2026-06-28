# MILP node-efficiency bench (issue #331, Step 1)

SCIP version: `10.0` · discopt config: `prod` defaults

## Reproduce — discopt vs SCIP (production config)

| instance | discopt nodes | discopt wall | SCIP nodes | SCIP wall | node ratio | obj match |
|---|---|---|---|---|---|---|
| mdk30x5 | 87 | 0.013s | 1 | 0.099s | 87.0x | True |
| mdk40x5 | 147 | 0.009s | 1 | 0.013s | 147.0x | True |
| mdk50x8 | 61 | 0.009s | 1 | 0.020s | 61.0x | True |
| mdk60x8 | 497 | 0.046s | 27 | 0.228s | 18.4x | True |
| mdk70x10 | 123 | 0.033s | 1 | 0.238s | 123.0x | True |
| mdk90x12 | 597 | 0.068s | 66 | 0.267s | 9.0x | True |
| mdk120x15 | 8387 | 0.527s | 2330 | 3.470s | 3.6x | True |
| mdk150x20 | 2335 | 0.446s | 579 | 1.374s | 4.0x | True |
| mdk200x25 | 27117 | 3.769s | 12688 | 16.409s | 2.1x | True |

`*` = hit per-solve time/node cap (not proven optimal); `obj match = n/a` when a solver did not prove optimality.


## Attribute — root bound & integrality gap closed

| instance | z_LP | z_opt | discopt root bound | discopt gap closed | SCIP gap closed |
|---|---|---|---|---|---|
| mdk30x5 | -921.41 | -902.00 | -912.57 | 45.5% | 83.8% |
| mdk40x5 | -1633.46 | -1620.00 | -1628.99 | 33.2% | 100.0% |
| mdk50x8 | -1769.45 | -1759.14 | -1764.04 | 52.5% | 101.4% |
| mdk60x8 | -2300.14 | -2276.16 | -2294.19 | 24.8% | 39.3% |
| mdk70x10 | -2349.74 | -2336.21 | -2346.38 | 24.9% | 65.7% |
| mdk90x12 | -3350.98 | -3329.33 | -3347.92 | 14.2% | 32.6% |
| mdk120x15 | -4638.87 | -4614.42 | -4636.58 | 9.4% | 19.9% |
| mdk150x20 | -5655.69 | -5635.56 | -5653.02 | 13.3% | 24.8% |
| mdk200x25 | -7428.07 | -7397.73 | -7426.55 | 5.0% | 10.9% |

## Ablation — node count per lever (one lever on top of baseline)

| instance | baseline | presolve | root_cuts | cut_rounds | node_cuts | strong_branch | heuristics | prod | full |
|---|---|---|---|---|---|---|---|---|---|
| mdk30x5 | 1319 | 1319 | 1415 | 1565 | 1451 | 1461 | 147 | 87 | 59 |
| mdk40x5 | 1547 | 1547 | 1287 | 769 | 1027 | 1671 | 191 | 147 | 65 |
| mdk50x8 | 131 | 131 | 641 | 35 | 255 | 259 | 41 | 61 | 35 |
| mdk60x8 | 1037 | 1037 | 1221 | 1309 | 1101 | 1997 | 605 | 497 | 215 |
| mdk70x10 | 4607 | 4607 | 4607 | 3457 | 4863 | 4869 | 195 | 123 | 77 |
| mdk90x12 | 2311 | 2311 | 2551 | 2757 | 2479 | 2673 | 673 | 597 | 435 |
| mdk120x15 | 20735 | 20735 | 20585 | 21503 | 19841 | 16895 | 10299 | 8387 | 7913 |
| mdk150x20 | 10751 | 10751 | 10623 | 11007 | 10239 | 10879 | 3125 | 2335 | 2333 |
| mdk200x25 | 58885 | 58885 | 57855 | 57229 | 57855 | 57089 | 29451 | 27117 | 28051 |

`*` = config hit node/time limit (not proven optimal). For capped rows the node count is *nodes explored within the budget*, not nodes-to-proof, so it conflates node-efficiency with per-node cost (stronger cuts make each node more expensive); read attribution primarily from the uncapped rows and cross-check wall time.

## Ablation — node reduction attributable to each lever (vs baseline)

| instance | presolve | root_cuts | cut_rounds | node_cuts | strong_branch | heuristics | prod | full |
|---|---|---|---|---|---|---|---|---|
| mdk30x5 | 0% | -7% | -19% | -10% | -11% | 89% | 93% | 96% |
| mdk40x5 | 0% | 17% | 50% | 34% | -8% | 88% | 90% | 96% |
| mdk50x8 | 0% | -389% | 73% | -95% | -98% | 69% | 53% | 73% |
| mdk60x8 | 0% | -18% | -26% | -6% | -93% | 42% | 52% | 79% |
| mdk70x10 | 0% | 0% | 25% | -6% | -6% | 96% | 97% | 98% |
| mdk90x12 | 0% | -10% | -19% | -7% | -16% | 71% | 74% | 81% |
| mdk120x15 | 0% | 1% | -4% | 4% | 19% | 50% | 60% | 62% |
| mdk150x20 | 0% | 1% | -2% | 5% | -1% | 71% | 78% | 78% |
| mdk200x25 | 0% | 2% | 3% | 2% | 3% | 50% | 54% | 52% |
