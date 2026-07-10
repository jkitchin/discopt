"""Exp C: paper-scale probe — 1-30-30-1 softplus NN (1021 weights), 3 trajectories.

Matrix-shaped NN emission (probe B pattern), shared weights across three
DAEBuilder blocks with different initial conditions. Full-space local solve.
Does the decomposition pain from the paper reproduce here?
"""

import time

import discopt.modeling as dm
import numpy as np
from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt.dae import ContinuousSet, DAEBuilder
from discopt.modeling import Model
from discopt.solvers.nlp_pounce import solve_nlp
from discopt.warm_start import validate_initial_solution
from hybrid_common import TF, make_data, r_true
from scipy.integrate import solve_ivp

rng = np.random.default_rng(0)
NH = 30
CA0S = [1.0, 0.8, 0.6]
NFE, NCP = 20, 2
REG = 1e-4

m = Model()
W1 = m.continuous("W1", shape=(1, NH), lb=-8, ub=8)
b1 = m.continuous("b1", shape=(NH,), lb=-8, ub=8)
W2 = m.continuous("W2", shape=(NH, NH), lb=-8, ub=8)
b2 = m.continuous("b2", shape=(NH,), lb=-8, ub=8)
W3 = m.continuous("W3", shape=(NH, 1), lb=-8, ub=8)
b3 = m.continuous("b3", lb=-8, ub=8)
n_weights = sum(v.size for v in (W1, b1, W2, b2, W3)) + 1


def nn_rate(c):
    """(nfe, ncp) expression -> (nfe, ncp) rate via a 1-30-30-1 softplus net."""
    h1 = dm.softplus(c[:, :, None] @ W1 + b1)  # (nfe, ncp, NH)
    h2 = dm.softplus(h1 @ W2 + b2)  # (nfe, ncp, NH)
    return (h2 @ W3)[:, :, 0] + b3  # (nfe, ncp)


obj = None
init = {}
builders = []
for k, ca0 in enumerate(CA0S):
    t_data, yA, yB = make_data(rng, ca0=ca0, cb0=0.0)
    cs = ContinuousSet(f"t{k}", (0.0, TF), nfe=NFE, ncp=NCP, scheme="radau")
    dae = DAEBuilder(m, cs)
    dae.add_state("cA", bounds=(0.0, 1.5), initial=ca0)
    dae.add_state("cB", bounds=(0.0, 1.5), initial=0.0)
    dae.set_ode(lambda t, s, a, u: {"cA": -nn_rate(s["cA"]), "cB": nn_rate(s["cA"])})
    dvars = dae.discretize()
    builders.append((dae, t_data, yA, yB, ca0))

    lsq = dae.least_squares("cA", t_data, yA) + dae.least_squares("cB", t_data, yB)
    obj = lsq if obj is None else obj + lsq

    tp = dae._element_points()
    init[dvars["cA"]] = np.interp(tp, np.concatenate([[0], t_data]), np.concatenate([[ca0], yA]))
    init[dvars["cB"]] = np.interp(tp, np.concatenate([[0], t_data]), np.concatenate([[0.0], yB]))

reg = REG * (dm.sum(W1**2) + dm.sum(b1**2) + dm.sum(W2**2) + dm.sum(b2**2) + dm.sum(W3**2) + b3**2)
m.minimize(obj + reg)

# Glorot-ish small random init for weights
init[W1] = rng.standard_normal((1, NH)) * 0.8
init[b1] = rng.standard_normal(NH) * 0.3
init[W2] = rng.standard_normal((NH, NH)) * (0.8 / np.sqrt(NH))
init[b2] = rng.standard_normal(NH) * 0.3
init[W3] = rng.standard_normal((NH, 1)) * (0.8 / np.sqrt(NH))
init[b3] = 0.0
x0 = validate_initial_solution(m, init)

n_vars = sum(v.size for v in m._variables)
print(f"problem: {n_vars} vars = {n_weights} weights + {n_vars - n_weights} states; 3 trajectories")

t0 = time.perf_counter()
ev = NLPEvaluator(m, gauss_newton=True)
t_build = time.perf_counter() - t0
print(f"evaluator build: {t_build:.1f}s  (gauss_newton fired: {ev.is_gauss_newton})")

t0 = time.perf_counter()
res = solve_nlp(ev, x0, options={"max_iter": 3000, "tol": 1e-8})
t_solve = time.perf_counter() - t0
print(
    f"solve: {res.status.name}  iters={res.iterations}  wall={t_solve:.1f}s  "
    f"obj={res.objective:.6e}"
)

# assess: recovered rate law + honest resimulation per trajectory
from discopt._jax.dag_compiler import _compute_var_offset  # noqa: E402


def value_of(var, x):
    off = _compute_var_offset(var, m)
    return np.asarray(x[off : off + var.size]).reshape(var.shape) if var.shape else x[off]


w1, bb1, w2, bb2, w3, bb3 = (value_of(v, res.x) for v in (W1, b1, W2, b2, W3, b3))


def softplus_np(z):
    return np.logaddexp(0.0, z)


def r_hat(c):
    h1 = softplus_np(np.atleast_1d(c)[:, None] @ w1 + bb1)
    h2 = softplus_np(h1 @ w2 + bb2)
    return (h2 @ w3)[:, 0] + bb3


c_grid = np.linspace(0.05, 1.0, 100)
rate_rmse = float(np.sqrt(np.mean((r_hat(c_grid) - r_true(c_grid)) ** 2)))
print(f"rate law RMSE over [0.05,1]: {rate_rmse:.4f}  (||r_true||_rms = 0.67)")

for dae, t_data, yA, yB, ca0 in builders:
    truth = solve_ivp(
        lambda t, y: [-r_true(y[0]), r_true(y[0])],
        (0, TF),
        [ca0, 0.0],
        t_eval=np.linspace(0, TF, 50),
        rtol=1e-10,
        atol=1e-12,
    )
    learned = solve_ivp(
        lambda t, y: [-r_hat(y[0]).item(), r_hat(y[0]).item()],
        (0, TF),
        [ca0, 0.0],
        t_eval=truth.t,
        rtol=1e-9,
        atol=1e-11,
    )
    rmse = float(np.sqrt(np.mean((learned.y[0] - truth.y[0]) ** 2)))
    print(f"  resim cA0={ca0}: RMSE vs truth = {rmse:.5f}")
