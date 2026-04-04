"""Adversary test: Markowitz 3-asset portfolio QP
Source: Markowitz (1952), analytically solvable. Covariance matrix and
        expected returns from Cornuejols & Tutuncu, "Optimization Methods
        in Finance", Example 3.1.
Known optimal: 0.08133 (minimum variance for target return >= 0.10)
Problem class: QP
"""
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import discopt.modeling as dm

# 3-asset portfolio
# Expected returns
mu = [0.08, 0.10, 0.12]
# Covariance matrix (symmetric positive definite)
# S = [[0.04, 0.006, 0.002],
#      [0.006, 0.09, 0.018],
#      [0.002, 0.018, 0.16]]
S = [[0.04, 0.006, 0.002], [0.006, 0.09, 0.018], [0.002, 0.018, 0.16]]
target_return = 0.10

m = dm.Model("markowitz")
w1 = m.continuous("w1", lb=0, ub=1)
w2 = m.continuous("w2", lb=0, ub=1)
w3 = m.continuous("w3", lb=0, ub=1)

# Minimize portfolio variance: w^T S w
variance = (
    S[0][0] * w1 * w1
    + 2 * S[0][1] * w1 * w2
    + 2 * S[0][2] * w1 * w3
    + S[1][1] * w2 * w2
    + 2 * S[1][2] * w2 * w3
    + S[2][2] * w3 * w3
)
m.minimize(variance)

# Budget constraint: weights sum to 1
m.subject_to(w1 + w2 + w3 == 1)

# Return constraint: expected return >= target
m.subject_to(mu[0] * w1 + mu[1] * w2 + mu[2] * w3 >= target_return)

# Solve with scipy to get known optimal
import numpy as np
from scipy.optimize import minimize as sp_min

S_np = np.array(S)
mu_np = np.array(mu)


def obj(w):
    return w @ S_np @ w


res = sp_min(
    obj,
    [1 / 3, 1 / 3, 1 / 3],
    method="SLSQP",
    bounds=[(0, 1)] * 3,
    constraints=[
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: mu_np @ w - target_return},
    ],
)
KNOWN_OPTIMAL = res.fun
print(f"Scipy reference: obj={KNOWN_OPTIMAL:.8f} w={res.x}")

# --- Solve with all three backends ---
backends = ["ipm", "ipopt", "ripopt"]
results = {}

for backend in backends:
    try:
        r = m.solve(time_limit=120, nlp_solver=backend)
        results[backend] = r
        rel_err = (
            abs(r.objective - KNOWN_OPTIMAL) / max(1, abs(KNOWN_OPTIMAL))
            if r.objective is not None
            else float("inf")
        )
        print(
            f"{backend:>6}: obj={r.objective}  "
            f"time={r.wall_time:.3f}s  "
            f"nodes={r.node_count}  "
            f"status={r.status}  "
            f"rel_err={rel_err:.2e}"
        )
    except Exception as e:
        print(f"{backend:>6}: FAILED - {e}")
        results[backend] = None

# --- Summary ---
print(f"\nKnown optimal: {KNOWN_OPTIMAL:.8f}")
for backend, r in results.items():
    if r is not None and r.objective is not None:
        rel_err = abs(r.objective - KNOWN_OPTIMAL) / max(
            1, abs(KNOWN_OPTIMAL)
        )
        print(f"{backend:>6}: {'PASS' if rel_err < 1e-3 else 'FAIL'}")
    else:
        print(f"{backend:>6}: NO SOLUTION")
