# MILP node-efficiency bench (issue #331, Step 1)

SCIP version: `10.0` · discopt config: `prod` defaults

## Reproduce — discopt vs SCIP (production config)

| instance | discopt nodes | discopt wall | SCIP nodes | SCIP wall | node ratio | obj match |
|---|---|---|---|---|---|---|
| mdk30x5 | 59 | 0.006s | 1 | 0.128s | 59.0x | True |
| mdk40x5 | 65 | 0.004s | 1 | 0.017s | 65.0x | True |
| mdk50x8 | 97 | 0.009s | 1 | 0.024s | 97.0x | True |
| mdk60x8 | 621 | 0.028s | 27 | 0.315s | 23.0x | True |
| mdk70x10 | 73 | 0.012s | 1 | 0.266s | 73.0x | True |
| mdk90x12 | 281 | 0.038s | 66 | 0.381s | 4.3x | True |
| mdk120x15 | 5113 | 0.243s | 2330 | 4.961s | 2.2x | True |
| mdk150x20 | 1725 | 0.173s | 579 | 1.932s | 3.0x | True |
| mdk200x25 | 18213 | 1.987s | 12688 | 23.630s | 1.4x | True |
| smdk50x15 | 955 | 0.085s | 21 | 0.235s | 45.5x | True |
| smdk60x20 | 1189 | 0.171s | 43 | 0.548s | 27.7x | True |
| smdk70x20 | 6183 | 0.589s | 145 | 1.274s | 42.6x | True |
| smdk80x25 | 40575 | 4.712s | 753 | 2.520s | 53.9x | True |

`*` = hit per-solve time/node cap (not proven optimal); `obj match = n/a` when a solver did not prove optimality.


## Attribute — root bound & integrality gap closed

| instance | z_LP | z_opt | discopt root bound | discopt gap closed | SCIP gap closed |
|---|---|---|---|---|---|
| mdk30x5 | -921.41 | -902.00 | -912.57 | 45.5% | 83.8% |
| mdk40x5 | -1633.46 | -1620.00 | -1628.99 | 33.2% | 100.0% |
| mdk50x8 | -1769.45 | -1759.00 | -1764.04 | 51.8% | 100.0% |
| mdk60x8 | -2300.14 | -2276.00 | -2294.19 | 24.6% | 39.0% |
| mdk70x10 | -2349.74 | -2336.00 | -2346.38 | 24.5% | 64.7% |
| mdk90x12 | -3350.98 | -3329.00 | -3347.92 | 14.0% | 32.1% |
| mdk120x15 | -4638.87 | -4614.37 | -4636.58 | 9.3% | 19.9% |
| mdk150x20 | -5655.69 | -5635.55 | -5653.02 | 13.3% | 24.8% |
| mdk200x25 | -7428.07 | -7397.70 | -7426.55 | 5.0% | 10.9% |
| smdk50x15 | -1510.38 | -1420.00 | -1483.43 | 29.8% | 81.4% |
| smdk60x20 | -2437.03 | -2328.00 | -2405.57 | 28.9% | 81.2% |
| smdk70x20 | -2858.66 | -2756.06 | -2847.27 | 11.1% | 62.9% |
| smdk80x25 | -3074.33 | -2959.29 | -3056.86 | 15.2% | 51.9% |

## Ablation — node count per lever (one lever on top of baseline)

| instance | baseline | presolve | root_cuts | cut_rounds | node_cuts | strong_branch | heuristics | reduced_cost_fixing | prod | full |
|---|---|---|---|---|---|---|---|---|---|---|
| mdk30x5 | 1319 | 1319 | 1441 | 1565 | 1451 | 1225 | 147 | 89 | 59 | 37 |
| mdk40x5 | 1547 | 1547 | 1287 | 769 | 1027 | 1745 | 191 | 31 | 65 | 37 |
| mdk50x8 | 131 | 131 | 641 | 35 | 255 | 387 | 41 | 41 | 97 | 91 |
| mdk60x8 | 1037 | 1037 | 1221 | 1309 | 1101 | 1531 | 605 | 257 | 621 | 161 |
| mdk70x10 | 4607 | 4607 | 4607 | 3457 | 4863 | 4675 | 195 | 91 | 73 | 51 |
| mdk90x12 | 2311 | 2311 | 2551 | 2757 | 2479 | 3297 | 673 | 315 | 281 | 501 |
| mdk120x15 | 20735 | 20735 | 20585 | 21503 | 19841 | 17151 | 10299 | 6331 | 5113 | 5183 |
| mdk150x20 | 10751 | 10751 | 10623 | 11007 | 10239 | 9855 | 3125 | 1973 | 1725 | 1781 |
| mdk200x25 | 58885 | 58885 | 57855 | 57229 | 57855 | 55303 | 29451 | 20235 | 18213 | 20565 |
| smdk50x15 | 17643 | 17643 | 3359 | 3071 | 517 | 4147 | 9193 | 7239 | 955 | 89 |
| smdk60x20 | 12125 | 12125 | 3603 | 2559 | 395 | 9973 | 6339 | 4741 | 1189 | 269 |
| smdk70x20 | 24961 | 24961 | 32471 | 10233 | 2303 | 19759 | 12733 | 11717 | 6183 | 365 |
| smdk80x25 | 124497 | 124497 | 154101 | 92463 | 6783 | 76631 | 67033 | 72531 | 40575 | 1207 |

`*` = config hit node/time limit (not proven optimal). For capped rows the node count is *nodes explored within the budget*, not nodes-to-proof, so it conflates node-efficiency with per-node cost (stronger cuts make each node more expensive); read attribution primarily from the uncapped rows and cross-check wall time.

## Ablation — node reduction attributable to each lever (vs baseline)

| instance | presolve | root_cuts | cut_rounds | node_cuts | strong_branch | heuristics | reduced_cost_fixing | prod | full |
|---|---|---|---|---|---|---|---|---|---|
| mdk30x5 | 0% | -9% | -19% | -10% | 7% | 89% | 93% | 96% | 97% |
| mdk40x5 | 0% | 17% | 50% | 34% | -13% | 88% | 98% | 96% | 98% |
| mdk50x8 | 0% | -389% | 73% | -95% | -195% | 69% | 69% | 26% | 31% |
| mdk60x8 | 0% | -18% | -26% | -6% | -48% | 42% | 75% | 40% | 84% |
| mdk70x10 | 0% | 0% | 25% | -6% | -1% | 96% | 98% | 98% | 99% |
| mdk90x12 | 0% | -10% | -19% | -7% | -43% | 71% | 86% | 88% | 78% |
| mdk120x15 | 0% | 1% | -4% | 4% | 17% | 50% | 69% | 75% | 75% |
| mdk150x20 | 0% | 1% | -2% | 5% | 8% | 71% | 82% | 84% | 83% |
| mdk200x25 | 0% | 2% | 3% | 2% | 6% | 50% | 66% | 69% | 65% |
| smdk50x15 | 0% | 81% | 83% | 97% | 76% | 48% | 59% | 95% | 99% |
| smdk60x20 | 0% | 70% | 79% | 97% | 18% | 48% | 61% | 90% | 98% |
| smdk70x20 | 0% | -30% | 59% | 91% | 21% | 49% | 53% | 75% | 99% |
| smdk80x25 | 0% | -24% | 26% | 95% | 38% | 46% | 42% | 67% | 99% |
