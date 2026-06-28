# MILP node-efficiency bench (issue #331, Step 1)

SCIP version: `10.0` · discopt config: `prod` defaults

## Reproduce — discopt vs SCIP (production config)

| instance | discopt nodes | discopt wall | SCIP nodes | SCIP wall | node ratio | obj match |
|---|---|---|---|---|---|---|
| mdk30x5 | 47 | 0.007s | 1 | 0.093s | 47.0x | True |
| mdk40x5 | 63 | 0.005s | 1 | 0.012s | 63.0x | True |
| mdk50x8 | 61 | 0.008s | 1 | 0.018s | 61.0x | True |
| mdk60x8 | 213 | 0.024s | 27 | 0.230s | 7.9x | True |
| mdk70x10 | 57 | 0.012s | 1 | 0.201s | 57.0x | True |
| mdk90x12 | 197 | 0.028s | 66 | 0.271s | 3.0x | True |
| mdk120x15 | 4713 | 0.276s | 2330 | 3.451s | 2.0x | True |
| mdk150x20 | 1797 | 0.329s | 579 | 1.443s | 3.1x | True |
| mdk200x25 | 17935 | 2.110s | 12688 | 16.264s | 1.4x | True |
| smdk50x15 | 703 | 0.120s | 21 | 0.192s | 33.5x | True |
| smdk60x20 | 1025 | 0.317s | 43 | 0.413s | 23.8x | True |
| smdk70x20 | 7643 | 0.936s | 145 | 0.944s | 52.7x | True |
| smdk80x25 | 29147 | 3.747s | 753 | 1.854s | 38.7x | True |

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
| smdk50x15 | -1510.38 | -1420.00 | -1483.43 | 29.8% | 81.4% |
| smdk60x20 | -2437.03 | -2328.00 | -2405.57 | 28.9% | 81.2% |
| smdk70x20 | -2858.66 | -2756.21 | -2847.27 | 11.1% | 62.9% |
| smdk80x25 | -3074.33 | -2959.27 | -3056.86 | 15.2% | 51.9% |

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
| smdk50x15 | 17643 | 17643 | 4397 | 3257 | 517 | 5197 | 9193 | 7239 | 703 | 183 |
| smdk60x20 | 12125 | 12125 | 4223 | 2047 | 395 | 8395 | 6339 | 4741 | 1025 | 377 |
| smdk70x20 | 24961 | 24961 | 32471 | 10725 | 2303 | 25495 | 12733 | 11717 | 7643 | 1183 |
| smdk80x25 | 124497 | 124497 | 162313 | 92435 | 6783 | 105297 | 67033 | 72531 | 29147 | 1885 |

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
| smdk50x15 | 0% | 75% | 82% | 97% | 71% | 48% | 59% | 96% | 99% |
| smdk60x20 | 0% | 65% | 83% | 97% | 31% | 48% | 61% | 92% | 97% |
| smdk70x20 | 0% | -30% | 57% | 91% | -2% | 49% | 53% | 69% | 95% |
| smdk80x25 | 0% | -30% | 26% | 95% | 15% | 46% | 42% | 77% | 98% |
