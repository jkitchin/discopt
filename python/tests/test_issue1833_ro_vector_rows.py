"""#1833: a vector-valued uncertain row is split by its *evaluated* shape.

A vector row ``body(x, p) <= rhs`` is one scalar row per element, and the
adversary picks a different worst case for each. ``RobustCounterpart`` splits
vector rows before any counterpart is built, but it read the row's shape from a
static inference and treated "unknown" as "scalar". ``x + M @ p <= 10`` was the
reported case (fixed by #1832's static matmul rule); ``exp(M @ p)`` and every
other composition the static rule did not cover still passed through whole,
and the polyhedral counterpart gave the vector ONE set of duals and reported a
feasible model infeasible.

The audit of the same path found two more ways an uncertain row reached a
counterpart that does not model it:

* the polyhedral counterpart takes the coefficient of ``p_j`` as the secant
  ``f(p_bar + e_j) - f(p_bar)``, exact only for a row affine in ``p``;
  ``x + exp(p) <= 30`` was certified at x = 17.94 (truth 30 - e^3 = 9.91);
* an uncertain parameter passed to an opaque ``dm.custom`` function was invisible
  to the RO walkers, so ``x + f(p) <= 10`` was solved at the nominal ``p``.

All three now refuse. Every robust optimum below is compared against an
independent oracle: vertex enumeration for box and polytope sets (exact for a
row affine in ``p``, and for one convex in ``p``) and the closed-form worst case
``a_i^T p_bar + rho ||Sigma^1/2 a_i||`` per scalar row for the ellipsoid.
"""

from __future__ import annotations

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import FunctionCall, SumOverExpression
from discopt.ro import (
    BoxUncertaintySet,
    EllipsoidalUncertaintySet,
    PolyhedralUncertaintySet,
    RobustCounterpart,
)
from discopt.ro.counterpart import _evaluated_shape
from discopt.ro.formulations.polyhedral import PolyhedralRobustFormulation

pytestmark = pytest.mark.smoke

M = np.array([[3.0, -1.0], [-2.0, 1.0]])
C = np.array([2.0, -1.0])
C22 = np.array([[1.0, -2.0], [0.5, 3.0]])
P1 = np.array([1.0, 1.0])
P2 = np.ones((2, 2))


def _vertices(A, b):
    k = A.shape[1]
    out: list[np.ndarray] = []
    for rows in itertools.combinations(range(len(b)), k):
        S = A[list(rows)]
        if abs(np.linalg.det(S)) < 1e-12:
            continue
        v = np.linalg.solve(S, b[list(rows)])
        if np.all(A @ v <= b + 1e-9) and not any(np.allclose(v, w) for w in out):
            out.append(v)
    return out


def _polytope(k):
    # |xi_j| <= 1 cut by the budget |sum xi| <= 1.2, which couples the components.
    A = np.vstack([np.eye(k), -np.eye(k), np.ones((1, k)), -np.ones((1, k))])
    b = np.concatenate([np.ones(2 * k), [1.2, 1.2]])
    return A, b


def _vertices_of(kind, pbar):
    k = pbar.size
    if kind == "box":
        delta = 0.5 * np.ones(k)
        return [
            pbar + (delta * np.array(s)).reshape(pbar.shape)
            for s in itertools.product((-1.0, 1.0), repeat=k)
        ]
    A, b = _polytope(k)
    verts = [pbar + v.reshape(pbar.shape) for v in _vertices(A, b)]
    assert len(verts) >= 2 * k
    return verts


def _uncertainty_set(kind, p, pbar):
    if kind == "box":
        return BoxUncertaintySet(p, delta=0.5 * np.ones(pbar.shape))
    A, b = _polytope(pbar.size)
    return PolyhedralUncertaintySet(p, A=A, b=b)


def _vars(m):
    x = m.continuous("x", lb=0, ub=20)
    X = m.continuous("X", shape=(2,), lb=0, ub=5)
    t = m.continuous("t", lb=0, ub=1)
    m.maximize(x + 0.1 * dm.sum(X) + 0.01 * t)
    return x, X, t


def _npsum(a, axis=None):
    return np.sum(a, axis=axis)


def _con(g, sense):
    return {"<=": lambda: g <= 0.0, ">=": lambda: g >= 0.0, "==": lambda: g == 0.0}[sense]()


# Each case: (p_bar, rows(p, x, X, t, S) -> list of ``(g, sense)`` for the rows
# ``g sense 0``). ``S`` is ``dm.sum`` in the robust model and ``np.sum`` in the
# vertex oracle and the feasibility check.
CASES = {
    "matmul_M_p": (P1, lambda p, x, X, t, S: [(x + M @ p - 10.0, "<=")]),
    "matmul_p_M": (P1, lambda p, x, X, t, S: [(x + p @ M - 10.0, "<=")]),
    "broadcast_c_times_p": (P1, lambda p, x, X, t, S: [(x + C * p - 10.0, "<=")]),
    "vector_var_plus_M_p": (P1, lambda p, x, X, t, S: [(X + M @ p - 6.0, "<=")]),
    "param_2d_elementwise": (P2, lambda p, x, X, t, S: [(x + p * C22 - 10.0, "<=")]),
    "param_2d_matmul": (P2, lambda p, x, X, t, S: [(x + p @ C - 10.0, "<=")]),
    "param_2d_axis_sum": (P2, lambda p, x, X, t, S: [(x + S(p * C22, axis=0) - 10.0, "<=")]),
    "ge_row": (P1, lambda p, x, X, t, S: [(10.0 - M @ p - x, ">=")]),
    "eq_row_variable_coeff": (
        P1,
        lambda p, x, X, t, S: [(x + (1 - t) * (M @ p) - (1 - t) * (M @ P1) - 4.0, "==")],
    ),
    "variable_coeff": (P1, lambda p, x, X, t, S: [(x + X * (M @ p) - 10.0, "<=")]),
    "sum_over_vector_terms": (
        P1,
        lambda p, x, X, t, S: [
            (x + (SumOverExpression([M @ p, C * p]) if S is dm.sum else M @ p + C * p) - 10.0, "<=")
        ],
    ),
}


def _oracle(case, kind):
    pbar, rows = CASES[case]
    m = dm.Model("oracle")
    x, X, t = _vars(m)
    verts = _vertices_of(kind, pbar)
    for v in verts:
        for g, sense in rows(v, x, X, t, _npsum):
            m.subject_to(_con(g, sense))
    r = m.solve()
    assert r.status == "optimal"
    return float(r.objective)


@pytest.mark.parametrize("kind", ["box", "polyhedral"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_vector_row_matches_vertex_oracle(case, kind):
    pbar, rows = CASES[case]
    m = dm.Model("ro")
    x, X, t = _vars(m)
    p = m.parameter("p", value=pbar.copy())
    for g, sense in rows(p, x, X, t, dm.sum):
        m.subject_to(_con(g, sense))
    unc, verts = _uncertainty_set(kind, p, pbar), _vertices_of(kind, pbar)
    RobustCounterpart(m, unc).formulate()
    r = m.solve()
    assert r.status == "optimal"
    # The returned point holds at every vertex (robust feasibility) ...
    xv, Xv, tv = float(r.value(x)), np.asarray(r.value(X)), float(r.value(t))
    n_checked = 0
    for v in verts:
        for g, sense in rows(v, xv, Xv, tv, _npsum):
            g = np.asarray(g, dtype=float)
            viol = {"<=": g, ">=": -g, "==": np.abs(g)}[sense]
            assert np.all(viol <= 1e-6), (case, kind, v, g)
            n_checked += g.size
    assert n_checked >= 2 * len(verts)
    # ... and is optimal against the vertex oracle (not conservative).
    assert r.objective == pytest.approx(_oracle(case, kind), abs=1e-5)


def test_issue_repro_matmul_row_polyhedral():
    """The reported model: x + M @ p <= 10 over |p_i - 1| <= 1 has x = 4."""
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=np.array([1.0, 1.0]))
    m.subject_to(x + M @ p <= 10)
    m.maximize(x)
    unc = PolyhedralUncertaintySet(p, A=np.vstack([np.eye(2), -np.eye(2)]), b=np.ones(4))
    RobustCounterpart(m, unc).formulate()
    r = m.solve()
    assert r.status == "optimal"
    assert r.objective == pytest.approx(4.0, abs=1e-6)


SIGMA = np.diag([1.0, 0.25])


@pytest.mark.parametrize(
    "case, row, lin",
    [
        ("matmul_M_p", lambda p, x: x + M @ p <= 10.0, lambda pv: M @ pv),
        ("ge_row", lambda p, x: 10.0 - M @ p >= x, lambda pv: M @ pv),
    ],
)
def test_ellipsoidal_vector_row_is_robust(case, row, lin):
    """The ellipsoidal answer is never above the closed-form robust optimum.

    Only soundness is asserted: the ellipsoidal matmul penalty is currently
    conservative on these rows (#1835, PR #1840), which is a separate defect.
    """
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=P1.copy())
    m.subject_to(row(p, x))
    m.maximize(x)
    RobustCounterpart(m, EllipsoidalUncertaintySet(p, rho=1.0, Sigma=SIGMA)).formulate()
    r = m.solve()
    assert r.status == "optimal"
    base = lin(P1)
    grads = np.stack([lin(P1 + e) - base for e in np.eye(2)], axis=1)
    worst = base + np.linalg.norm(grads @ np.linalg.cholesky(SIGMA), axis=1)
    truth = min(20.0, 10.0 - float(worst.max()))
    assert r.objective <= truth + 1e-6


# -- shape evaluation ---------------------------------------------------------


@pytest.mark.requires_jax
def test_evaluated_shape_covers_compositions_static_inference_misses():
    m = dm.Model()
    x = m.continuous("x")
    p = m.parameter("p", value=P1.copy())
    P = m.parameter("P", value=P2.copy())
    assert _evaluated_shape(x + dm.exp(0.1 * (M @ p))) == (2,)
    assert _evaluated_shape(x + dm.maximum(M @ p, 0.0)) == (2,)
    assert _evaluated_shape(x + SumOverExpression([M @ p, C * p])) == (2,)
    assert _evaluated_shape(x + P @ C) == (2,)
    assert _evaluated_shape(x + dm.sum(P * C22, axis=1)) == (2,)
    assert _evaluated_shape(x + P * C22) == (2, 2)
    assert _evaluated_shape(x + dm.sum(M @ p)) == ()
    assert _evaluated_shape(x + (M @ p)[1]) == ()
    # A node with no numpy rule is unknown, never scalar.
    assert _evaluated_shape(x + dm.custom(lambda a: a)(p)) is None


def test_unknown_shape_uncertain_row_is_refused():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=np.array(1.0))
    # A function node with no shape rule: its shape is unknown, never scalar.
    m.subject_to(x + FunctionCall("no_such_op", p) <= 10)
    m.maximize(x)
    with pytest.raises(NotImplementedError, match="cannot determine the shape"):
        RobustCounterpart(m, BoxUncertaintySet(p, delta=1.0)).formulate()


# -- polyhedral refusals ------------------------------------------------------


def test_polyhedral_function_of_vector_row_refused_not_infeasible():
    """``exp(M @ p)`` passed through unsplit and was reported infeasible."""
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=40)
    p = m.parameter("p", value=P1.copy())
    m.subject_to(x + dm.exp(0.1 * (M @ p)) <= 10)
    m.maximize(x)
    unc = PolyhedralUncertaintySet(p, A=np.vstack([np.eye(2), -np.eye(2)]), b=np.ones(4))
    with pytest.raises(NotImplementedError, match="exact only for a row affine"):
        RobustCounterpart(m, unc).formulate()


@pytest.mark.parametrize(
    "term",
    [
        pytest.param(lambda p: dm.exp(p), id="exp_p"),
        pytest.param(lambda p: p * p, id="p_times_p"),
        pytest.param(lambda p: 1.0 / p, id="one_over_p"),
        pytest.param(lambda p: abs(p), id="abs_p"),
        pytest.param(lambda p: p**2, id="p_squared"),
        pytest.param(lambda p: 2.0 * (p * (p + 1.0)), id="nested_product"),
    ],
)
def test_polyhedral_nonlinear_in_p_refused(term):
    """#1841: the secant coefficient ``f(p+e_j)-f(p)`` is exact only for affine rows.

    ``x + exp(p) <= 30`` over ``|p - 1| <= 2`` was solved to x = 17.94 (truth
    9.91): a non-robust answer reported as robust.
    """
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=40)
    p = m.parameter("p", value=np.array(1.0))
    m.subject_to(x + term(p) <= 30)
    m.maximize(x)
    unc = PolyhedralUncertaintySet(p, A=np.array([[1.0], [-1.0]]), b=np.array([2.0, 2.0]))
    with pytest.raises(NotImplementedError, match="exact only for a row affine"):
        RobustCounterpart(m, unc).formulate()


@pytest.mark.parametrize(
    "objective",
    [lambda x, p: x * p, lambda x, p: x - 2.0 * p / 4.0, lambda x, p: -(x + p)],
)
def test_polyhedral_affine_objective_still_accepted(objective):
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=4)
    p = m.parameter("p", value=np.array(1.0))
    m.maximize(objective(x, p))
    unc = PolyhedralUncertaintySet(p, A=np.array([[1.0], [-1.0]]), b=np.array([0.5, 0.5]))
    RobustCounterpart(m, unc).formulate()
    assert m.solve().status == "optimal"


def test_polyhedral_formulation_refuses_unsplit_vector_row():
    """Driven directly, the formulation refuses a vector row instead of dualizing it."""
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=P1.copy())
    m.subject_to(x + M @ p <= 10)
    m.maximize(x)
    unc = PolyhedralUncertaintySet(p, A=np.vstack([np.eye(2), -np.eye(2)]), b=np.ones(4))
    with pytest.raises(NotImplementedError, match="not one scalar row"):
        PolyhedralRobustFormulation(m, [unc]).build()


@pytest.mark.requires_jax
@pytest.mark.parametrize("kind", ["box", "polyhedral", "ellipsoidal"])
@pytest.mark.parametrize("where", ["constraint", "objective", "nested"])
def test_custom_call_of_uncertain_param_is_refused(kind, where):
    """``x + f(p) <= 10`` with an opaque ``f`` was solved at p_bar (x = 9; truth 8).

    No counterpart can see inside ``f``, so every set must refuse -- detecting
    ``p`` is not enough, since substitution cannot reach it either (#1847).
    """
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=np.array([1.0]))
    f = dm.custom(lambda a: a[0] * 1.0)(p)
    if where == "constraint":
        m.subject_to(x + f <= 10)
        m.maximize(x)
    elif where == "objective":
        m.subject_to(x <= 10)
        m.maximize(x - f)
    else:
        m.subject_to(x + 2.0 * dm.exp(0.1 * f) <= 10)
        m.maximize(x)
    unc = {
        "box": lambda: BoxUncertaintySet(p, delta=np.array([1.0])),
        "polyhedral": lambda: PolyhedralUncertaintySet(
            p, A=np.array([[1.0], [-1.0]]), b=np.array([1.0, 1.0])
        ),
        "ellipsoidal": lambda: EllipsoidalUncertaintySet(p, rho=1.0),
    }[kind]()
    with pytest.raises(NotImplementedError, match="opaque custom function"):
        RobustCounterpart(m, unc).formulate()


def test_split_broadcasts_array_rhs():
    """A scalar body against an array rhs is one scalar row per rhs element."""
    from discopt.modeling.core import Constraint
    from discopt.ro.counterpart import _split_vector_uncertain_rows

    m = dm.Model()
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=np.array(1.0))
    con = Constraint(body=x + p, sense="<=", rhs=np.array([5.0, 1.0]))
    out = _split_vector_uncertain_rows([con], {"p"})
    assert [c.rhs for c in out] == [5.0, 1.0]
    assert all(_evaluated_shape(c.body) == () for c in out)
