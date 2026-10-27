"""#1975 step 1, global: min over the scaled limit of (cap - P), solved by SCIP."""
from pyscipopt import Model, quicksum
m = Model(); m.hideOutput()
a = m.addVar("a", lb=0, ub=1); u0 = m.addVar("u0", lb=0); u1 = m.addVar("u1", lb=0); u6 = m.addVar("u6", lb=0)
y18 = m.addVar("y18", lb=0); y19 = m.addVar("y19", lb=0)
x = {k: m.addVar(f"x{k}", lb=0) for k in [2, 3, 4, 5] + list(range(9, 18))}
m.addCons(u0 * x[2] + u1 * x[4] <= y18); m.addCons(u0 * x[3] + u1 * x[5] <= y19)
m.addCons(y18 <= 3340.8 * a + 500 * (1 - a)); m.addCons(y19 <= 371.2 * a + 4500 * (1 - a))
m.addCons(356.474947137148 * u0 + 53.7083537310174 * u1 <= 0.564264890180399 * y18)
m.addCons(339.983422262764 * u0 + 43.5418249774113 * u1 <= 0.405939876920766 * y19)
m.addCons(106.946746119538 * u0 + 145.018955433089 * u1 <= 0.507117039797071 * y18)
m.addCons(173.929713444361 * u0 + 203.031384299627 * u1 <= 0.578889145413521 * y19)
for c0, c1, c6, r in [(0.94, -0.048, -0.064, [(1, 2), (1.2, 1.8), (0.8, 2.2)]),
                      (0.0, -0.247, 0.58, [(2, 3), (1.8, 3.5), (2.4, 2.3)]),
                      (-0.244, 0.916, -0.172, [(3, 3), (2.7, 3.2), (3.2, 2.7)])]:
    for c7, c8 in r:
        m.addCons(c0 * u0 + c1 * u1 + c6 * u6 <= c7 * a + c8 * (1 - a))
S1 = quicksum(x[k] for k in (9, 10, 11)); S2 = quicksum(x[k] for k in (12, 13, 14)); S3 = quicksum(x[k] for k in (15, 16, 17))
m.addCons(x[2] + x[3] <= 0.94 * S1 - 0.244 * S3)
m.addCons(0.064 * S1 - 0.58 * S2 + 0.172 * S3 <= 0)
m.addCons(x[4] + x[5] + 0.048 * S1 + 0.247 * S2 - 0.916 * S3 <= 0)
xs = [x[k] for k in range(9, 18)]
m.addCons(quicksum(c * v for c, v in zip([1, 1.2, .8, 2, 1.8, 2.4, 3, 2.7, 3.2], xs)) <= 3712)
m.addCons(quicksum(c * v for c, v in zip([2, 1.8, 2.2, 3, 3.5, 2.3, 3, 3.2, 2.7], xs)) <= 5000)
gap = m.addVar("gap", lb=None)
m.addCons(gap == 3712 * a + 5000 * (1 - a) - (u0 * (x[2] + x[3]) + u1 * (x[4] + x[5])))
m.setObjective(gap, "minimize"); m.setParam("limits/gap", 0); m.setParam("limits/absgap", 1e-9)
m.optimize()
print(m.getStatus(), "primal", m.getObjVal(), "dual", m.getDualbound())
print({v.name: round(m.getVal(v), 4) for v in [a, u0, u1, u6]})
