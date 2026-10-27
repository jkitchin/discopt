"""Same transformed outer piece as prolog_projective.py, solved by SCIP (cross-check)."""
import sys
from pyscipopt import Model, quicksum
M = float(sys.argv[1])
m = Model(); m.hideOutput()
s = m.addVar("s", lb=0, ub=1 / M)
u = {k: m.addVar(f"u{k}", lb=0) for k in (0, 1, 6, 7, 8, 18, 19)}
x = {k: m.addVar(f"x{k}", lb=0) for k in [2, 3, 4, 5] + list(range(9, 18))}
c = m.addCons
c(u[7] + u[8] == 1)
for k in (0, 1, 6): c(u[k] >= 0.2 * s)
c(x[2] * u[0] + x[4] * u[1] - u[18] <= 0); c(x[3] * u[0] + x[5] * u[1] - u[19] <= 0)
S1 = x[9] + x[10] + x[11]; S2 = x[12] + x[13] + x[14]; S3 = x[15] + x[16] + x[17]
c(x[2] + x[3] - 0.94 * S1 + 0.244 * S3 <= 0); c(0.064 * S1 - 0.58 * S2 + 0.172 * S3 <= 0)
c(x[4] + x[5] + 0.048 * S1 + 0.247 * S2 - 0.916 * S3 <= 0)
xs = [x[k] for k in range(9, 18)]
c(quicksum(a * v for a, v in zip([1, 1.2, .8, 2, 1.8, 2.4, 3, 2.7, 3.2], xs)) <= 3712)
c(quicksum(a * v for a, v in zip([2, 1.8, 2.2, 3, 3.5, 2.3, 3, 3.2, 2.7], xs)) <= 5000)
c(356.474947137148 * u[0] + 53.7083537310174 * u[1] + s * x[2] - 0.564264890180399 * u[18] <= 352 * s)
c(339.983422262764 * u[0] + 43.5418249774113 * u[1] + s * x[3] - 0.405939876920766 * u[19] <= 430 * s)
c(106.946746119538 * u[0] + 145.018955433089 * u[1] + s * x[4] - 0.507117039797071 * u[18] <= 222 * s)
c(173.929713444361 * u[0] + 203.031384299627 * u[1] + s * x[5] - 0.578889145413521 * u[19] <= 292 * s)
c(u[18] - 3340.8 * u[7] - 500 * u[8] <= 0); c(u[19] - 371.2 * u[7] - 4500 * u[8] <= 0)
for c0, c1, c6, rr in [(.94, -.048, -.064, [(1, 2), (1.2, 1.8), (.8, 2.2)]),
                       (0.0, -.247, .58, [(2, 3), (1.8, 3.5), (2.4, 2.3)]),
                       (-.244, .916, -.172, [(3, 3), (2.7, 3.2), (3.2, 2.7)])]:
    for c7, c8 in rr: c(c0 * u[0] + c1 * u[1] + c6 * u[6] - c7 * u[7] - c8 * u[8] <= 0)
g = m.addVar("g", lb=None)
c(g == -(x[2] * u[0] + x[3] * u[0] + x[4] * u[1] + x[5] * u[1]) + 3712 * u[7] + 5000 * u[8])
m.setObjective(g); m.setParam("limits/time", 120); m.setParam("limits/absgap", 1e-4); m.setParam("limits/gap", 0)
m.optimize()
print(f"SCIP M={M:g}: {m.getStatus()} primal {m.getPrimalbound():.6g} dual {m.getDualbound():.6g} "
      f"nodes {m.getNNodes()} time {m.getSolvingTime():.1f}s")
