"""Linear-equality constraint-factor RLT (``DISCOPT_RLT_LINEQ``).

For a linear equality ``a'x + c == 0`` and any variable ``x_j``, the product
``(a'x + c) * x_j == 0`` is an identity on the feasible region; linearized over
the lifted product columns it is ``sum_i a_i X_ij + c x_j == 0``. That is
Sherali–Adams level 1 restricted to the *equality* factors — the one RLT family
that needs neither a binary variable (unlike ``_relax/rlt.py``'s exhaustive
RLT-1 LP, whose ``X_ii = x_i`` diagonal forces binaries) nor the node box
(unlike the bound-factor families).

Why it matters: on continuous nonconvex QPs the McCormick-only root bound is
hopeless, and this family is where the recoverable gap lives. Measured on QPLIB
(root LP vs published optimum): QPLIB_1157 −13.874 → −11.245 against −10.948
(89.9 % of the gap), QPLIB_1493 49.7 %, QPLIB_1143 32.2 %, QPLIB_1423 29.0 %,
QPLIB_1507 27.5 %; the bound-factor rows added nothing on top.

These tests lock the three properties the pass is built on:

1. **Soundness** — no feasible point is cut. Checked directly at the lifted
   point ``(x, X = x x')`` of sampled feasible ``x``, against every row of the
   relaxation (not only the new ones).
2. **Tightening, without crossing** — the ON bound is ``>=`` the OFF bound and
   still ``<=`` the true optimum of the sampled/known problem.
3. **Box-independence** — the rows the pass emits are byte-identical at two
   different boxes. The native spatial kernel regenerates only per-term envelope
   rows when it patches a node box and carries everything else forward
   unchanged, so a box-dependent row here would go stale and unsound at depth.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt._relax.uniform_relax import build_uniform_relaxation
from scipy.optimize import linprog

pytestmark = [pytest.mark.relaxation, pytest.mark.correctness]


def _stqp(n: int, seed: int) -> tuple[dm.Model, np.ndarray]:
    """Standard quadratic program ``min x'Qx  s.t.  sum(x) == 1, 0 <= x <= 1``.

    Indefinite ``Q`` (a symmetric uniform draw), so the McCormick relaxation of
    the objective is genuinely loose. Generic structure, not a named instance.
    """
    rng = np.random.default_rng(seed)
    A = rng.uniform(-1.0, 1.0, size=(n, n))
    Q = 0.5 * (A + A.T)
    m = dm.Model()
    x = [m.continuous(f"x{i}", lb=0.0, ub=1.0) for i in range(n)]
    m.minimize(sum(Q[i, j] * x[i] * x[j] for i in range(n) for j in range(n)))
    m.subject_to(sum(x) == 1.0)
    return m, Q


def _root_lp(model, lb, ub, *, rlt_lineq):
    """Root LP bound of the uniform relaxation, plus the LP data itself.

    Solved with HiGHS rather than the in-house simplex purely for test runtime;
    the bound is a property of the relaxation, not of the LP engine.
    """
    rel = build_uniform_relaxation(model, box=(lb, ub), rlt_lineq=rlt_lineq)
    M = rel.model
    A = sp.csr_matrix(M._A_ub)
    b = np.asarray(M._b_ub, dtype=float).ravel()
    bnds = [
        (float(lo) if np.isfinite(lo) else None, float(hi) if np.isfinite(hi) else None)
        for lo, hi in np.asarray(M._bounds, dtype=float)
    ]
    res = linprog(
        np.asarray(M._c, dtype=float).ravel(), A_ub=A, b_ub=b, bounds=bnds, method="highs"
    )
    assert res.status == 0, res.message
    return float(res.fun), rel, A, b


def _lifted_point(rel, x: np.ndarray) -> np.ndarray:
    """The exact lifted point for a feasible ``x``: originals, then every
    registered product column set to the true product of its factors.

    Columns the maps do not name (other aux families) are left at 0 — such a
    column is not touched by an RLT row, so it cannot mask a violation there.
    """
    z = np.zeros(len(rel.model._bounds), dtype=float)
    z[: rel.n_orig] = x
    for (a, b), col in rel.bilinear_map.items():
        z[col] = x[a] * x[b]
    for (i, p), col in rel.monomial_map.items():
        z[col] = x[i] ** p
    return z


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_feasible_point_is_cut(seed):
    """Every row of the RLT-on relaxation holds at the exact lifted image of a
    feasible point. This is the soundness property; an RLT row derived wrongly
    (e.g. dropping the constraint's ``rhs`` shift) fails here."""
    n = 6
    model, _Q = _stqp(n, seed)
    lb = np.zeros(n)
    ub = np.ones(n)
    rel = build_uniform_relaxation(model, box=(lb, ub), rlt_lineq=True)
    A = sp.csr_matrix(rel.model._A_ub)
    b = np.asarray(rel.model._b_ub, dtype=float).ravel()

    rng = np.random.default_rng(1000 + seed)
    checked = 0
    for _ in range(25):
        w = rng.dirichlet(np.ones(n))  # feasible: w >= 0, sum(w) == 1
        z = _lifted_point(rel, w)
        resid = A @ z - b
        assert np.max(resid) <= 1e-7, f"feasible point cut by {np.max(resid):.3e}"
        checked += 1
    assert checked == 25


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_bound_tightens_and_never_crosses_the_optimum(seed):
    """ON is at least as tight as OFF, and neither crosses the true optimum.

    The true optimum is bracketed from above by the best sampled feasible
    objective; a bound above that would be a false certificate.
    """
    n = 6
    model, Q = _stqp(n, seed)
    lb = np.zeros(n)
    ub = np.ones(n)

    off, _rel_off, _A0, _b0 = _root_lp(model, lb, ub, rlt_lineq=False)
    on, _rel_on, _A1, _b1 = _root_lp(model, lb, ub, rlt_lineq=True)

    # Upper bound on the true optimum from sampled feasible points (vertices of
    # the simplex included — an StQP optimum is often on a face).
    rng = np.random.default_rng(seed)
    best = min(
        float(w @ Q @ w) for w in [*np.eye(n), *(rng.dirichlet(np.ones(n)) for _ in range(400))]
    )

    assert on >= off - 1e-7, f"RLT loosened the bound: {on} < {off}"
    assert on <= best + 1e-7, f"FALSE BOUND: {on} > feasible objective {best}"
    assert off <= best + 1e-7


def test_rows_are_box_independent():
    """The emitted RLT rows are identical at two different boxes.

    ``spatial_producer`` reads structure off a probe box and fixed rows off the
    real box, and the Rust kernel re-derives only per-term envelope rows at each
    node. A box-dependent row would silently go stale down the tree.
    """
    n = 5
    model, _Q = _stqp(n, seed=7)

    def rlt_rows(lo, hi):
        base = build_uniform_relaxation(model, box=(lo, hi), rlt_lineq=False)
        full = build_uniform_relaxation(model, box=(lo, hi), rlt_lineq=True)
        A_b = sp.csr_matrix(base.model._A_ub)
        A_f = sp.csr_matrix(full.model._A_ub)
        assert A_f.shape[0] > A_b.shape[0], "the RLT pass emitted no rows"
        extra = A_f[A_b.shape[0] :].toarray()
        return extra, np.asarray(full.model._b_ub, dtype=float).ravel()[A_b.shape[0] :]

    a1, r1 = rlt_rows(np.zeros(n), np.ones(n))
    a2, r2 = rlt_rows(np.full(n, 0.1), np.full(n, 0.8))
    assert a1.shape == a2.shape
    assert np.array_equal(a1, a2), "RLT rows changed with the box"
    assert np.array_equal(r1, r2)


def test_equality_with_a_nonzero_constant_is_shifted_correctly():
    """``a'x == 3`` must give ``sum_i a_i X_ij - 3 x_j == 0``, not ``sum_i a_i X_ij == 0``.

    Dropping the constant makes the row *wrong* rather than merely weak, and it
    cuts the feasible region — caught here by evaluating every row at the exact
    lifted image of feasible points.

    Note where the constant lives: :class:`Constraint` normalizes to ``body == 0``
    and *enforces* ``rhs == 0``, so the ``-3`` arrives inside the body's affine
    constant, and that is the term under test. The pass also subtracts
    ``constraint.rhs`` (mirroring the base constraint-row loop), which is a no-op
    under that contract and is not what this test exercises.
    """
    m = dm.Model()
    x = [m.continuous(f"x{i}", lb=0.0, ub=4.0) for i in range(3)]
    m.minimize(x[0] * x[1] - x[1] * x[2] + x[0] * x[2])
    m.subject_to(x[0] + 2.0 * x[1] + x[2] == 3.0)

    lb = np.zeros(3)
    ub = np.full(3, 4.0)
    rel = build_uniform_relaxation(m, box=(lb, ub), rlt_lineq=True)
    A = sp.csr_matrix(rel.model._A_ub)
    b = np.asarray(rel.model._b_ub, dtype=float).ravel()

    checked = 0
    rng = np.random.default_rng(3)
    for _ in range(50):
        x0 = rng.uniform(0.0, 3.0)
        x2 = rng.uniform(0.0, min(4.0, 3.0 - x0)) if x0 < 3.0 else 0.0
        x1 = (3.0 - x0 - x2) / 2.0
        w = np.array([x0, x1, x2])
        assert abs(w[0] + 2 * w[1] + w[2] - 3.0) < 1e-12
        if np.any(w < -1e-12) or np.any(w > 4.0 + 1e-12):
            continue
        resid = A @ _lifted_point(rel, w) - b
        assert np.max(resid) <= 1e-7, f"feasible point cut by {np.max(resid):.3e}"
        checked += 1
    assert checked >= 40, f"only {checked} feasible samples exercised the rows"
