"""Exp E2: certified global training attempt with a SCALARIZED collocation model.

Same GP/RBF hybrid as exp E, but collocation constraints emitted one scalar
equation at a time (bypassing DAEBuilder's vectorized emission) so the global
machinery (convexity classifier / spatial B&B / AMP) sees the scalar shapes it
was built for.
"""

import time

import discopt.modeling as dm
import numpy as np
from discopt.dae.polynomials import collocation_matrix, radau_roots
from discopt.modeling import Model
from discopt.solvers.nlp_pounce import solve_nlp_from_model
from discopt.warm_start import validate_initial_solution
from hybrid_common import CA0, TF, make_data

rng = np.random.default_rng(0)
t_data, yA, yB = make_data(rng)

NC, ELL, REG = 4, 0.3, 1e-4
centers = np.linspace(0.1, 1.0, NC)
NFE, NCP = 6, 2
A, _w = collocation_matrix(NCP, "radau")
h = TF / NFE
tp = np.zeros((NFE, NCP + 1))
eb = np.linspace(0.0, TF, NFE + 1)
cp = radau_roots(NCP)
for i in range(NFE):
    tp[i, 0] = eb[i]
    tp[i, 1:] = eb[i] + h * cp

m = Model()
alpha = m.continuous("alpha", shape=(NC,), lb=-5.0, ub=5.0)
# scalar state variables x[i][k], cA only (cB = CA0 - cA eliminated by conservation)
x = [[m.continuous(f"x_{i}_{k}", lb=0.0, ub=1.2) for k in range(NCP + 1)] for i in range(NFE)]
m.subject_to(x[0][0] == CA0)


def rate(c):
    out = None
    for j in range(NC):
        term = alpha[j] * dm.exp(-((c - float(centers[j])) ** 2) / (2 * ELL**2))
        out = term if out is None else out + term
    return out


for i in range(NFE):
    for j in range(1, NCP + 1):
        lhs = None
        for k in range(NCP + 1):
            t_ = float(A[j - 1, k]) * x[i][k]
            lhs = t_ if lhs is None else lhs + t_
        m.subject_to(lhs == -h * rate(x[i][j]))
for i in range(NFE - 1):
    m.subject_to(x[i + 1][0] == x[i][NCP])

# least squares: snap each measurement to nearest node (scalar objective)
tp_flat = tp.ravel()
obj = None
for t_i, ya_i, yb_i in zip(t_data, yA, yB):
    idx = int(np.argmin(np.abs(tp_flat - t_i)))
    e, k = idx // (NCP + 1), idx % (NCP + 1)
    ra = (x[e][k] - float(ya_i)) ** 2
    rb = ((CA0 - x[e][k]) - float(yb_i)) ** 2
    obj = ra + rb if obj is None else obj + ra + rb
for j in range(NC):
    obj = obj + REG * alpha[j] ** 2
m.minimize(obj)

# local warm start
init = {}
for i in range(NFE):
    for k in range(NCP + 1):
        init[x[i][k]] = float(
            np.interp(tp[i, k], np.concatenate([[0], t_data]), np.concatenate([[CA0], yA]))
        )
init[alpha] = np.zeros(NC)
x0 = validate_initial_solution(m, init)
loc = solve_nlp_from_model(m, x0=x0, options={"max_iter": 3000, "tol": 1e-8})
print(f"local : {loc.status.name}  obj={loc.objective:.6e}")

from discopt._jax.dag_compiler import _compute_var_offset  # noqa: E402


def value_of(var, xs):
    off = _compute_var_offset(var, m)
    return np.asarray(xs[off : off + var.size]).reshape(var.shape) if var.shape else xs[off]


warm = {v: value_of(v, loc.x) for row in x for v in row}
warm[alpha] = value_of(alpha, loc.x)

n_vars = sum(v.size for v in m._variables)
print(f"scalarized model: {n_vars} vars, {len(m._constraints)} constraints")

t0 = time.perf_counter()
res = m.solve(time_limit=300, gap_tolerance=0.10, initial_solution=warm)
wall = time.perf_counter() - t0
print(f"global: status={res.status} obj={res.objective}")
print(
    f"  bound={res.bound} gap={res.gap} nodes={res.node_count} "
    f"certified={res.gap_certified} root_bound={res.root_bound} wall={wall:.1f}s"
)
