"""#875: end-to-end overrun on the issue's SHAPE, at a scale the defect bites.

``watercontamination0202`` (106,711 vars / 107,209 rows / 7 binaries) is not
available in this environment, so this reproduces its shape — very many variables,
very many linear rows, a small nonlinear core, a handful of binaries — and measures
``solve(time_limit=T)`` overrun end to end. That is the issue's own acceptance
criterion (``wall <= 1.25 * T``) applied to a model that can actually be built here.

Run on both sides of the change to compare.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402


def build(n: int) -> dm.Model:
    m = dm.Model("wide")
    x = m.continuous("x", shape=(n,), lb=-5.0, ub=5.0)
    b = m.binary("b", shape=(7,))
    # Many plain linear rows — these are what LinearContext assembles.
    for k in range(0, n, 2):
        m.subject_to(x[k] + x[k + 1] <= 3.0)
    for k in range(0, n, 4):
        m.subject_to(x[k] == 0.5)
    # Small nonlinear core.
    for k in range(0, 200, 2):
        m.subject_to(x[k] * x[k + 1] <= 4.0)
    m.subject_to(sum(b[j] for j in range(7)) == 3)
    m.minimize(sum(x[k] for k in range(200)) + sum(b))
    return m


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    for n in (8_000, 20_000):
        for T in (20.0,):
            m = build(n)
            t0 = time.perf_counter()
            res = m.solve(time_limit=T)
            wall = time.perf_counter() - t0
            print(
                f"[{label}] n_vars={n:>7,d} T={T:>5.0f}s  wall={wall:>7.1f}s "
                f"({wall / T:>5.2f}x)  status={res.status:<11s} nodes={res.node_count}",
                flush=True,
            )


if __name__ == "__main__":
    main()
