"""Entry experiment: train a neural-DAE with discopt's existing parts.

Hybrid physics+ML model in the style of Lueg et al. (arXiv:2504.04665):
  physics : batch reactor mass balances   dcA/dt = -r,  dcB/dt = +r
  ML      : unknown rate law r(cA) replaced by a small trainable NN
  method  : simultaneous — orthogonal collocation (Radau) via discopt.dae.DAEBuilder,
            NN weights and discretized states are joint decision variables,
            least-squares data mismatch + L2 regularization, one local NLP solve.

No framework changes: DAEBuilder + array Variables + dm.tanh + solve_nlp_from_model.
"""

import discopt.modeling as dm
import numpy as np
from discopt.dae import ContinuousSet, DAEBuilder
from discopt.modeling import Model
from discopt.solvers.nlp_pounce import solve_nlp_from_model
from discopt.warm_start import validate_initial_solution
from scipy.integrate import solve_ivp

rng = np.random.default_rng(0)

# ---------------------------------------------------------------- ground truth
CA0, CB0, TF = 1.0, 0.0, 2.0


def r_true(c):
    # "unknown" nonlinear kinetics the NN must recover (Michaelis-Menten-like)
    return 1.5 * c**2 / (0.3 + c)


def rhs_true(t, y):
    r = r_true(y[0])
    return [-r, r]


t_data = np.linspace(0.05, TF, 15)
sol = solve_ivp(rhs_true, (0, TF), [CA0, CB0], t_eval=t_data, rtol=1e-10, atol=1e-12)
noise = 0.01
yA = sol.y[0] + noise * rng.standard_normal(len(t_data))
yB = sol.y[1] + noise * rng.standard_normal(len(t_data))

# ---------------------------------------------------------------- hybrid model
NC = 12  # kernel centers
ELL = 0.12  # fixed lengthscale
REG = 1e-4
centers = np.linspace(0.0, 1.05, NC)

m = Model()
alpha = m.continuous("alpha", shape=(NC,), lb=-50.0, ub=50.0)


def nn_rate(c):
    """GP posterior mean / RBF expansion: linear in the trainable weights alpha."""
    out = None
    for j in range(NC):
        term = alpha[j] * dm.exp(-((c - float(centers[j])) ** 2) / (2 * ELL**2))
        out = term if out is None else out + term
    return out


cs = ContinuousSet("t", (0.0, TF), nfe=20, ncp=2, scheme="radau")
dae = DAEBuilder(m, cs)
dae.add_state("cA", bounds=(0.0, 1.5), initial=CA0)
dae.add_state("cB", bounds=(0.0, 1.5), initial=CB0)


def rhs(t, s, a, u):
    r = nn_rate(s["cA"])
    return {"cA": -r, "cB": r}


dae.set_ode(rhs)
dvars = dae.discretize()

lsq = dae.least_squares("cA", t_data, yA) + dae.least_squares("cB", t_data, yB)
reg = REG * dm.sum(alpha**2)
m.minimize(lsq + reg)

# ---------------------------------------------------------------- warm start
# states: interpolate the noisy data onto the collocation grid (paper-style init);
# weights: small random.
tp = dae._element_points()  # (nfe, ncp+1)
cA_init = np.interp(tp, np.concatenate([[0], t_data]), np.concatenate([[CA0], yA]))
cB_init = np.interp(tp, np.concatenate([[0], t_data]), np.concatenate([[CB0], yB]))

x0 = validate_initial_solution(
    m,
    {
        alpha: np.zeros(NC),
        dvars["cA"]: cA_init,
        dvars["cB"]: cB_init,
    },
)

# ---------------------------------------------------------------- train = solve one NLP
res = solve_nlp_from_model(m, x0=x0, options={"max_iter": 3000, "tol": 1e-8})
print(f"status      : {res.status}  iters={res.iterations}  wall={res.wall_time:.2f}s")
print(f"objective   : {res.objective:.6e}")

# ---------------------------------------------------------------- assess
from discopt._jax.dag_compiler import _compute_var_offset  # noqa: E402


def value_of(var, x):
    off = _compute_var_offset(var, m)
    return np.asarray(x[off : off + var.size]).reshape(var.shape) if var.shape else x[off]


AL = value_of(alpha, res.x)


def r_hat(c):
    K = np.exp(-((np.atleast_1d(c)[:, None] - centers[None, :]) ** 2) / (2 * ELL**2))
    return K @ AL


# 1) recovered rate law vs truth on the range visited by the data
c_grid = np.linspace(yA.min(), CA0, 100)
rate_rmse = float(np.sqrt(np.mean((r_hat(c_grid).ravel() - r_true(c_grid)) ** 2)))
rate_scale = float(np.sqrt(np.mean(r_true(c_grid) ** 2)))
print(f"rate law    : RMSE(r_hat - r_true) = {rate_rmse:.4f}  (||r_true||_rms = {rate_scale:.4f})")


# 2) collocation trajectory vs noiseless truth
class _Result:
    """Minimal SolveResult shim so extract_solution can read the NLP solution."""

    @staticmethod
    def value(v):
        return value_of(v, res.x)


t_traj, cA_traj = dae.extract_solution(_Result, "cA")
truth = solve_ivp(rhs_true, (0, TF), [CA0, CB0], t_eval=t_traj, rtol=1e-10, atol=1e-12)
traj_rmse = float(np.sqrt(np.mean((cA_traj - truth.y[0]) ** 2)))
print(f"trajectory  : RMSE(cA_colloc - cA_true) = {traj_rmse:.5f}  (noise level = {noise})")

# 3) honest generalization test: integrate the LEARNED ODE with scipy from t=0
learned = solve_ivp(
    lambda t, y: [-r_hat(y[0]).item(), r_hat(y[0]).item()],
    (0, TF),
    [CA0, CB0],
    t_eval=t_traj,
    rtol=1e-9,
    atol=1e-11,
)
sim_rmse = float(np.sqrt(np.mean((learned.y[0] - truth.y[0]) ** 2)))
print(f"resimulation: RMSE(cA_learned_ode - cA_true) = {sim_rmse:.5f}")

n_vars = sum(v.size for v in m._variables)
n_nn = NC
print(f"problem size: {n_vars} variables ({n_nn} NN weights + {n_vars - n_nn} collocation states)")
