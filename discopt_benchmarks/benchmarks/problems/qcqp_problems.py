"""Nonconvex QCQP benchmark problems with rigorously-known optima.

These instances are the *scoreboard* for Wave-2 relaxation-strength work (PSD /
eigenvalue cuts, SOC cuts): they exercise the gap that term-wise McCormick / RLT
relaxations leave on coupled bilinear structure, which is exactly what those cuts
close.

Two families, both box-constrained on ``[0, 1]^n``:

* **Concave BoxQP** (``Q`` negative definite) — a concave objective attains its
  global minimum at a *vertex* of the box, so the optimum is computed exactly by
  enumerating the ``2^n`` vertices. These are rigorous correctness anchors (their
  reference optimum needs no solver), though their convex-envelope relaxation is
  already tight at the root.
* **Indefinite QCQP** (``Q`` symmetric, indefinite) — the optimum generally lies
  on a face/interior and the relaxation has a genuine root gap, so the solver
  must branch. The reference optimum is found by dense multi-start local polish
  (a quadratic over a box is reliably solved to global optimality from a grid of
  starts at these sizes), and re-verified in the test suite.

Both ``Q`` and ``c`` are generated deterministically from a seed, so every
instance — and its hardcoded ``known_optimum`` — is reproducible and checkable
(``test_qcqp_scoreboard.py`` recomputes the references).
"""

from __future__ import annotations

import itertools

import discopt.modeling as dm
import numpy as np
from scipy.optimize import minimize as scipy_minimize

from benchmarks.problems.base import TestProblem, register

_APPLICABLE = ["pounce", "ipopt"]


# ────────────────────────────────────────────────────────────────
# Deterministic instance generators
# ────────────────────────────────────────────────────────────────


def concave_boxqp(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Negative-definite ``Q`` (concave objective) and linear term ``c``."""
    rng = np.random.default_rng(seed)
    L = rng.standard_normal((n, n))
    Q = -(L @ L.T) - 0.5 * np.eye(n)
    c = rng.standard_normal(n)
    return Q, c


def indefinite_qcqp(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric (generically indefinite) ``Q`` and linear term ``c``."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    Q = A + A.T
    c = rng.standard_normal(n)
    return Q, c


def build_boxqp_model(name: str, Q: np.ndarray, c: np.ndarray) -> dm.Model:
    """Build ``min 0.5 xᵀ Q x + cᵀ x`` over ``x ∈ [0, 1]^n`` as a discopt Model."""
    n = len(c)
    m = dm.Model(name)
    x = [m.continuous(f"x{i}", lb=0.0, ub=1.0) for i in range(n)]
    quad = dm.sum([0.5 * float(Q[i, j]) * x[i] * x[j] for i in range(n) for j in range(n)])
    lin = dm.sum([float(c[i]) * x[i] for i in range(n)])
    m.minimize(quad + lin)
    return m


# ────────────────────────────────────────────────────────────────
# Reference optima (used to set + verify known_optimum)
# ────────────────────────────────────────────────────────────────


def reference_optimum_vertex(Q: np.ndarray, c: np.ndarray) -> float:
    """Exact global minimum of a *concave* BoxQP via vertex enumeration."""
    n = len(c)
    best = np.inf
    for bits in itertools.product((0.0, 1.0), repeat=n):
        x = np.asarray(bits)
        best = min(best, float(0.5 * x @ Q @ x + c @ x))
    return best


def reference_optimum_multistart(Q: np.ndarray, c: np.ndarray, *, levels=(0.0, 0.5, 1.0)) -> float:
    """Global minimum of a BoxQP via dense multi-start L-BFGS-B polish."""
    n = len(c)
    bounds = [(0.0, 1.0)] * n

    def fun(x):
        return float(0.5 * x @ Q @ x + c @ x)

    def jac(x):
        return Q @ x + c

    best = np.inf
    for start in itertools.product(levels, repeat=n):
        res = scipy_minimize(
            fun, np.asarray(start, float), jac=jac, bounds=bounds, method="L-BFGS-B"
        )
        best = min(best, float(res.fun))
    return best


# ────────────────────────────────────────────────────────────────
# Registered instances
#
# known_optimum values are reproduced by reference_optimum_* above and
# re-checked in test_qcqp_scoreboard.py.
# ────────────────────────────────────────────────────────────────

# (name, n, seed, family, level, known_optimum, tags)
_INSTANCES = [
    # Concave anchors — rigorous vertex optima (correctness backstop).
    ("qcqp_concave_n6", 6, 6, "concave", "smoke", -27.7849929843, ["concave", "anchor"]),
    ("qcqp_concave_n5", 5, 5, "concave", "full", -21.6345403793, ["concave", "anchor"]),
    # Indefinite coupled-bilinear — genuine root gap; the solver must branch.
    ("qcqp_indef_n4_s1", 4, 1, "indefinite", "smoke", -1.3170272600, ["indefinite"]),
    ("qcqp_indef_n4_s2", 4, 2, "indefinite", "smoke", -3.0274212861, ["indefinite"]),
    ("qcqp_indef_n4_s3", 4, 3, "indefinite", "smoke", -3.3403661021, ["indefinite"]),
    ("qcqp_indef_n5_s11", 5, 11, "indefinite", "full", -6.3650050196, ["indefinite"]),
    ("qcqp_indef_n6_s12", 6, 12, "indefinite", "full", -4.1590036779, ["indefinite"]),
]


def _make_builder(family: str, n: int, seed: int, name: str):
    def _build() -> dm.Model:
        Q, c = concave_boxqp(n, seed) if family == "concave" else indefinite_qcqp(n, seed)
        return build_boxqp_model(name, Q, c)

    return _build


for _name, _n, _seed, _family, _level, _opt, _tags in _INSTANCES:
    register(
        TestProblem(
            name=_name,
            category="qcqp",
            level=_level,
            build_fn=_make_builder(_family, _n, _seed, _name),
            known_optimum=_opt,
            applicable_solvers=_APPLICABLE,
            n_vars=_n,
            n_constraints=0,
            tags=_tags,
        )
    )
