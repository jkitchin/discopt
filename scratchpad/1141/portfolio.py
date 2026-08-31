"""Cardinality-constrained Markowitz portfolio: the convex MIQCP class of
`portfol_classical050_1` (MINLPLib), rebuilt with the modeling API because the
MINLPLib snapshot is not reachable from this container.

min  -mu'x   s.t.  x'Sigma x <= v,  sum x = 1,  x_i <= b_i,  sum b <= K,  b binary.
Sigma = F F'/k + diag(d)  (PSD by construction, dense cross terms).
"""
import numpy as np
import discopt.modeling as dm


def build(n=20, K=5, seed=0, kfac=4, var_cap=None):
    rng = np.random.default_rng(seed)
    F = rng.normal(scale=0.1, size=(n, kfac))
    Sigma = F @ F.T / kfac + np.diag(0.01 + 0.02 * rng.random(n))
    mu = 0.05 + 0.15 * rng.random(n)
    m = dm.Model("portfolio")
    x = [m.continuous(f"x{i}", lb=0.0, ub=1.0) for i in range(n)]
    b = [m.binary(f"b{i}") for i in range(n)]
    quad = 0
    for i in range(n):
        for j in range(n):
            if abs(Sigma[i, j]) > 0:
                quad = quad + float(Sigma[i, j]) * x[i] * x[j]
    if var_cap is None:
        var_cap = float(np.mean(np.diag(Sigma)) / K * 1.5)
    m.subject_to(quad <= var_cap)
    m.subject_to(sum(x) == 1.0)
    for i in range(n):
        m.subject_to(x[i] - b[i] <= 0.0)
    m.subject_to(sum(b) <= K)
    m.minimize(-sum(float(mu[i]) * x[i] for i in range(n)))
    return m
