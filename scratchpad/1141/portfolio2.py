"""Degenerate cardinality-constrained Markowitz portfolio (the #1141 class).

`portfol_classical050_1`'s hardness is *binary degeneracy*: the objective barely
depends on which assets are chosen, so a combinatorial number of cardinality
assignments tie at the master LP optimum and OA removes them one at a time.
`spread` controls how much the expected returns differ; small spread reproduces
that regime.
"""
import numpy as np
import discopt.modeling as dm


def build(n=50, K=5, seed=0, kfac=4, spread=0.002, cap_scale=1.0):
    rng = np.random.default_rng(seed)
    F = rng.normal(scale=0.1, size=(n, kfac))
    Sigma = F @ F.T / kfac + np.diag(0.01 + 0.02 * rng.random(n))
    mu = 0.10 + spread * rng.random(n)
    m = dm.Model("portfolio")
    x = [m.continuous(f"x{i}", lb=0.0, ub=1.0) for i in range(n)]
    b = [m.binary(f"b{i}") for i in range(n)]
    quad = 0
    for i in range(n):
        for j in range(n):
            quad = quad + float(Sigma[i, j]) * x[i] * x[j]
    cap = cap_scale * float(np.mean(np.diag(Sigma)) / K)
    m.subject_to(quad <= cap)
    m.subject_to(sum(x) == 1.0)
    for i in range(n):
        m.subject_to(x[i] - b[i] <= 0.0)
    m.subject_to(sum(b) <= K)
    m.minimize(-sum(float(mu[i]) * x[i] for i in range(n)))
    return m
