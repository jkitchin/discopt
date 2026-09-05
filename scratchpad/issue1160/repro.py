import numpy as np
import discopt.modeling.core as dm
from discopt.validation.feasibility import verify_point

m = dm.Model("sum_axis")
A = m.continuous("A", shape=(2, 3), lb=0, ub=1)
m.subject_to(dm.sum(A, axis=1) <= 2)     # per-ROW cap
m.minimize(-dm.sum(A))                   # maximize the total

r = m.solve(time_limit=60, gap_tolerance=1e-6)
print("status", r.status, "obj", r.objective, "bound", r.bound)

better = np.array([1., 1., 0., 1., 1., 0.])
print(verify_point(m, better, with_objective=True))
