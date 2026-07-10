"""Exp E: certified GLOBAL training of the linear-in-alpha GP/RBF hybrid.

Small instance: 6 RBF centers, nfe=8, ncp=2 (54 vars). Spatial B&B via
Model.solve() with the local solution as initial incumbent. Question: does the
dual bound make progress, i.e. is certified-global training even plausible?
"""

import time

import discopt.modeling as dm
import numpy as np
from discopt.dae import ContinuousSet, DAEBuilder
from discopt.modeling import Model
from discopt.solvers.nlp_pounce import solve_nlp_from_model
from discopt.warm_start import validate_initial_solution
from hybrid_common import CA0, CB0, TF, make_data

rng = np.random.default_rng(0)
t_data, yA, yB = make_data(rng)

NC, ELL, REG = 6, 0.22, 1e-4
centers = np.linspace(0.0, 1.05, NC)
NFE, NCP = 8, 2


def build():
    m = Model()
    alpha = m.continuous("alpha", shape=(NC,), lb=-10.0, ub=10.0)

    def rate(c):
        out = None
        for j in range(NC):
            term = alpha[j] * dm.exp(-((c - float(centers[j])) ** 2) / (2 * ELL**2))
            out = term if out is None else out + term
        return out

    cs = ContinuousSet("t", (0.0, TF), nfe=NFE, ncp=NCP, scheme="radau")
    dae = DAEBuilder(m, cs)
    dae.add_state("cA", bounds=(0.0, 1.5), initial=CA0)
    dae.add_state("cB", bounds=(0.0, 1.5), initial=CB0)
    dae.set_ode(lambda t, s, a, u: {"cA": -rate(s["cA"]), "cB": rate(s["cA"])})
    dvars = dae.discretize()
    lsq = dae.least_squares("cA", t_data, yA) + dae.least_squares("cB", t_data, yB)
    m.minimize(lsq + REG * dm.sum(alpha**2))
    return m, dae, dvars, alpha


# local solve first (warm start / incumbent)
m, dae, dvars, alpha = build()
tp = dae._element_points()
init = {
    alpha: np.zeros(NC),
    dvars["cA"]: np.interp(tp, np.concatenate([[0], t_data]), np.concatenate([[CA0], yA])),
    dvars["cB"]: np.interp(tp, np.concatenate([[0], t_data]), np.concatenate([[CB0], yB])),
}
x0 = validate_initial_solution(m, init)
loc = solve_nlp_from_model(m, x0=x0, options={"max_iter": 3000, "tol": 1e-8})
print(f"local : {loc.status.name}  obj={loc.objective:.6e}")

# global attempt on a fresh model, warm-started with the local solution
from discopt._jax.dag_compiler import _compute_var_offset  # noqa: E402


def value_of(var, x, model):
    off = _compute_var_offset(var, model)
    return np.asarray(x[off : off + var.size]).reshape(var.shape) if var.shape else x[off]


m2, dae2, dvars2, alpha2 = build()
warm = {
    alpha2: value_of(alpha, loc.x, m),
    dvars2["cA"]: value_of(dvars["cA"], loc.x, m),
    dvars2["cB"]: value_of(dvars["cB"], loc.x, m),
}

t0 = time.perf_counter()
res = m2.solve(time_limit=180, gap_tolerance=0.10, initial_solution=warm)
wall = time.perf_counter() - t0
print(f"global: status={res.status}  obj={getattr(res, 'objective', None)}")
for attr in ("bound", "gap", "node_count", "nodes", "gap_certified", "convex_fast_path"):
    if hasattr(res, attr):
        print(f"  {attr} = {getattr(res, attr)}")
print(f"  wall = {wall:.1f}s")
