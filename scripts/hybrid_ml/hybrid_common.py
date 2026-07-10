"""Shared builder for the hybrid neural-DAE experiments (batch reactor A->B)."""

import discopt.modeling as dm
import numpy as np
from discopt.dae import ContinuousSet, DAEBuilder
from discopt.modeling import Model
from discopt.warm_start import validate_initial_solution
from scipy.integrate import solve_ivp

CA0, CB0, TF = 1.0, 0.0, 2.0


def r_true(c):
    return 1.5 * c**2 / (0.3 + c)


def make_data(rng, n_obs=15, noise=0.01, ca0=CA0, cb0=CB0):
    def rhs_true(t, y):
        r = r_true(y[0])
        return [-r, r]

    t_data = np.linspace(0.05, TF, n_obs)
    sol = solve_ivp(rhs_true, (0, TF), [ca0, cb0], t_eval=t_data, rtol=1e-10, atol=1e-12)
    yA = sol.y[0] + noise * rng.standard_normal(n_obs)
    yB = sol.y[1] + noise * rng.standard_normal(n_obs)
    return t_data, yA, yB


def build_nn_model(nh=6, nfe=20, ncp=2, reg=1e-4, seed=0, act=dm.tanh):
    """Single-trajectory hybrid model with a 1-nh-1 NN rate law. Returns everything."""
    rng = np.random.default_rng(seed)
    t_data, yA, yB = make_data(rng)

    m = Model()
    w1 = m.continuous("w1", shape=(nh,), lb=-8.0, ub=8.0)
    b1 = m.continuous("b1", shape=(nh,), lb=-8.0, ub=8.0)
    w2 = m.continuous("w2", shape=(nh,), lb=-8.0, ub=8.0)
    b2 = m.continuous("b2", lb=-8.0, ub=8.0)

    def nn_rate(c):
        out = b2
        for j in range(nh):
            out = out + w2[j] * act(w1[j] * c + b1[j])
        return out

    cs = ContinuousSet("t", (0.0, TF), nfe=nfe, ncp=ncp, scheme="radau")
    dae = DAEBuilder(m, cs)
    dae.add_state("cA", bounds=(0.0, 1.5), initial=CA0)
    dae.add_state("cB", bounds=(0.0, 1.5), initial=CB0)
    dae.set_ode(lambda t, s, a, u: {"cA": -nn_rate(s["cA"]), "cB": nn_rate(s["cA"])})
    dvars = dae.discretize()

    lsq = dae.least_squares("cA", t_data, yA) + dae.least_squares("cB", t_data, yB)
    m.minimize(lsq + reg * (dm.sum(w1**2) + dm.sum(b1**2) + dm.sum(w2**2) + b2**2))

    tp = dae._element_points()
    cA_init = np.interp(tp, np.concatenate([[0], t_data]), np.concatenate([[CA0], yA]))
    cB_init = np.interp(tp, np.concatenate([[0], t_data]), np.concatenate([[CB0], yB]))
    x0 = validate_initial_solution(
        m,
        {
            w1: 0.5 * rng.standard_normal(nh),
            b1: 0.5 * rng.standard_normal(nh),
            w2: 0.5 * rng.standard_normal(nh),
            b2: 0.0,
            dvars["cA"]: cA_init,
            dvars["cB"]: cB_init,
        },
    )
    return m, dae, x0, (t_data, yA, yB)
