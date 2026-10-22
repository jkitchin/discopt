"""#1720: a dense vectorized LP must not reach a dense extractor.

``dm.sum(C * x)`` with an ndarray ``C`` and ``dm.sum(x, axis=k) <= v`` over a
1000x1000 ``x`` declined every sparse LP-extraction rung (the tape on its node
budget), fell through to ``jax.jacobian`` and was SIGKILLed at ~38 GB. After
extraction, two post-solve consumers (row-matched duals, incumbent repair) hit
the JAX evaluator's dense ``jacfwd`` sparsity trace and its ``J^T J`` coloring.

The tests force the same route at small ``n`` by shrinking the tape's node
budget, so the regression runs in milliseconds; the arrays are tiny but the
code path is exactly the one the 1000x1000 model takes.
"""

from __future__ import annotations

import discopt._nl_expr_compiler as nlc
import pytest

pytest.importorskip("jax", reason="JAX is the optional discopt[jax] extra (#1873)")

import discopt._relax.nlp_evaluator as nle
import discopt._relax.problem_classifier as pc
import discopt.modeling as dm
import numpy as np
import scipy.sparse as sp
from discopt import Model
from discopt._relax.affine_array import NotAffineError, affine_of


def _dense(A):
    return A.toarray() if sp.issparse(A) else np.asarray(A)


def _transport(n: int, *, maximize: bool = False, seed: int = 0) -> Model:
    rng = np.random.default_rng(seed)
    a, b = rng.uniform(50, 100, n), rng.uniform(10, 40, n)
    c = rng.uniform(1, 10, (n, n))
    m = Model("transport")
    x = m.continuous("x", shape=(n, n), lb=0, ub=1e4)
    if maximize:
        m.maximize(dm.sum(-c * x))
    else:
        m.minimize(dm.sum(c * x))
    m.subject_to(dm.sum(x, axis=1) <= a, name="sup")
    m.subject_to(dm.sum(x, axis=0) >= b, name="dem")
    return m


def _mixed(n: int) -> Model:
    """Equalities, scalar and array bodies, matmul, slices and broadcasting."""
    rng = np.random.default_rng(1)
    m = Model("mixed")
    x = m.continuous("x", shape=(n, n), lb=-5, ub=5)
    y = m.continuous("y", shape=(n,), lb=-5, ub=5)
    z = m.continuous("z", lb=-5, ub=5)
    W = rng.normal(size=(n, n))
    m.minimize(dm.sum(W * x) - 2.0 * dm.sum(y) + 3.0 * z + 1.5)
    m.subject_to(dm.sum(x, axis=-1) - y == 0.5, name="bal")
    m.subject_to(W @ y + z <= 4.0, name="mv")
    m.subject_to(x[0, :] - x[:, 1] / 2.0 >= -1.0, name="sl")
    m.subject_to(dm.sum(x) + z <= 3.0, name="tot")
    m.subject_to(-(y[1:] - y[:-1]) <= 1.0, name="ramp")
    return m


def _assert_lp_equal(got, want):
    for f in ("c", "b_eq", "x_l", "x_u"):
        np.testing.assert_array_equal(np.asarray(getattr(got, f)), np.asarray(getattr(want, f)))
    np.testing.assert_array_equal(_dense(got.A_eq), _dense(want.A_eq))
    assert got.obj_const == want.obj_const
    assert list(got.row_sense) == list(want.row_sense)


# ---------------------------------------------------------------------------
# The walk itself: exact against numpy at random points.
# ---------------------------------------------------------------------------


def _walk_cases():
    rng = np.random.default_rng(7)
    m = Model("walk")
    X = m.continuous("X", shape=(3, 4), lb=-1, ub=1)
    v = m.continuous("v", shape=(4,), lb=-1, ub=1)
    s = m.continuous("s", lb=-1, ub=1)
    C = rng.normal(size=(3, 4))
    A = rng.normal(size=(2, 3))
    B = rng.normal(size=(4, 5))
    u = rng.normal(size=4)
    cases = [
        (C * X, lambda X, v, s: C * X),
        (dm.sum(C * X), lambda X, v, s: np.sum(C * X)),
        (dm.sum(X, axis=0), lambda X, v, s: X.sum(axis=0)),
        (dm.sum(X, axis=1), lambda X, v, s: X.sum(axis=1)),
        (dm.sum(X, axis=-2), lambda X, v, s: X.sum(axis=-2)),
        (X + v, lambda X, v, s: X + v),
        (X - s * 2.0, lambda X, v, s: X - 2.0 * s),
        (X[1, :] - v / 3.0, lambda X, v, s: X[1, :] - v / 3.0),
        (X[:, 2], lambda X, v, s: X[:, 2]),
        (X[0, 3], lambda X, v, s: X[0, 3]),
        (-X, lambda X, v, s: -X),
        (A @ X, lambda X, v, s: A @ X),
        (X @ B, lambda X, v, s: X @ B),
        (X @ u, lambda X, v, s: X @ u),
        (u @ v, lambda X, v, s: u @ v),
        (v @ B, lambda X, v, s: v @ B),
        (dm.sum([X[0, 0], 2.0 * v[1], s]), lambda X, v, s: X[0, 0] + 2 * v[1] + s),
        (dm.sum(dm.sum(X, axis=1)), lambda X, v, s: X.sum()),
        (np.ones((3, 4)) * 2.5 + X * C, lambda X, v, s: 2.5 + X * C),
    ]
    return m, (X, v, s), cases


def test_affine_walk_matches_numpy():
    m, (X, v, s), cases = _walk_cases()
    n = sum(var.size for var in m._variables)
    rng = np.random.default_rng(3)
    checked = 0
    for expr, ref in cases:
        aff = affine_of(expr, m._flat_var_offset, n)
        for _ in range(3):
            xf = rng.normal(size=n)
            Xv = xf[m._flat_var_offset(X) :][: X.size].reshape(X.shape)
            vv = xf[m._flat_var_offset(v) :][: v.size]
            sv = float(xf[m._flat_var_offset(s)])
            want = np.asarray(ref(Xv, vv, sv), dtype=np.float64)
            got = (aff.M @ xf if aff.M is not None else 0.0) + aff.b
            assert aff.shape == want.shape, expr
            np.testing.assert_allclose(got, want.reshape(-1), rtol=1e-12, atol=1e-12)
            checked += 1
    assert checked == 3 * len(cases)


@pytest.mark.parametrize(
    "build",
    [
        lambda X, v, s: X * X,
        lambda X, v, s: v @ v,
        lambda X, v, s: dm.exp(v),
        lambda X, v, s: 1.0 / (s + 2.0),
        lambda X, v, s: s**2,
    ],
)
def test_affine_walk_refuses_nonaffine(build):
    m, (X, v, s), _ = _walk_cases()
    n = sum(var.size for var in m._variables)
    with pytest.raises(NotAffineError):
        affine_of(build(X, v, s), m._flat_var_offset, n)


def test_affine_walk_parameter_policy():
    m = Model("p")
    x = m.continuous("x", shape=(3,), lb=0, ub=1)
    p = m.parameter("p", value=2.0)
    n = 3
    aff = affine_of(p * x, m._flat_var_offset, n)
    np.testing.assert_array_equal(aff.M.toarray(), 2.0 * np.eye(3))
    with pytest.raises(NotAffineError):
        affine_of(p * x, m._flat_var_offset, n, allow_parameters=False)


# ---------------------------------------------------------------------------
# The LP rung: identical to the tape rung wherever both accept.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build", [lambda: _transport(6), lambda: _transport(5, maximize=True), lambda: _mixed(4)]
)
@pytest.mark.parametrize("for_qp", [False, True])
def test_vectorized_rung_equals_tape_rung(build, for_qp):
    m = build()
    want = pc._extract_lp_data_tape(m, for_qp=for_qp)
    assert want is not None
    got = pc._extract_lp_data_vectorized(m, for_qp=for_qp)
    assert got is not None
    _assert_lp_equal(got, want)


def test_vectorized_rung_declines_nonlinear():
    m = Model("nl")
    x = m.continuous("x", shape=(3,), lb=0, ub=1)
    m.minimize(dm.sum(x))
    m.subject_to(x * x <= 0.5)
    assert pc._extract_lp_data_vectorized(m) is None


def test_vectorized_rung_declines_builder_rows():
    """Builder-resident rows (#840) live outside ``model._constraints``; a rung
    that walked only that list would drop them, so it must decline instead."""
    m = _transport(3)
    x = m._variables[0]
    m.add_linear_constraints(np.ones((1, x.size)), x, "<=", np.array([50.0]))
    assert m._builder_linear_blocks
    assert pc._extract_lp_data_vectorized(m) is None


def test_tape_decline_routes_to_vectorized_not_autodiff(monkeypatch):
    """The #1720 route: the tape declines on its node budget; the dense JAX rung
    must not be reached, and the result is what the tape gives when it can."""
    want = pc._extract_lp_data_tape(_transport(6))
    assert want is not None

    monkeypatch.setattr(nlc, "_MAX_TAPE_NODES", 10)

    def _boom(*_a, **_k):
        raise AssertionError("dense autodiff extractor reached")

    monkeypatch.setattr(pc, "_extract_lp_data_autodiff", _boom)
    m = _transport(6)
    assert pc._extract_lp_data_tape(m) is None  # the probe fired: tape declined
    _assert_lp_equal(pc.extract_lp_data(m), want)


def test_autodiff_refuses_loudly_beyond_dense_budget(monkeypatch):
    monkeypatch.setattr(pc, "_AUTODIFF_DENSE_MAX_BYTES", 8 * 10)
    with pytest.raises(pc.LPExtractionTooLargeError, match="dense Jacobian"):
        pc._extract_lp_data_autodiff(_transport(4))


# ---------------------------------------------------------------------------
# Post-solve consumers.
# ---------------------------------------------------------------------------


def test_affine_constraint_rows_match_evaluator():
    from discopt._tape_nlp_evaluator import make_evaluator
    from discopt.solver import _affine_constraint_rows, _csr_rows

    m = _mixed(4)
    n = sum(v.size for v in m._variables)
    J, g0, row_map = _affine_constraint_rows(m, n)
    ev = make_evaluator(m)
    x0 = np.zeros(n)
    np.testing.assert_allclose(
        J.toarray(), _csr_rows(ev.evaluate_sparse_jacobian(x0), n).toarray(), atol=0, rtol=1e-15
    )
    np.testing.assert_array_equal(g0, np.asarray(ev.evaluate_constraints(x0)).reshape(-1))
    assert [(a, b, c) for a, b, c in row_map] == list(ev.constraint_row_map())


def test_jax_evaluator_affine_pattern_matches_trace(monkeypatch):
    m = _mixed(4)
    traced = nle.NLPEvaluator(m)
    want = traced.sparsity_pattern.jacobian_sparsity.toarray()
    assert getattr(traced, "_affine_jacobian", None) is None  # small: probe path

    monkeypatch.setattr(nle, "_DENSE_JACOBIAN_COMPILE_LIMIT", 0)
    ev = nle.NLPEvaluator(m)
    got = ev.sparsity_pattern.jacobian_sparsity.toarray()
    assert ev._affine_jacobian is not None  # the probe fired: affine arm taken
    np.testing.assert_array_equal(got, want)
    x = np.random.default_rng(0).uniform(-1, 1, size=ev.n_variables)
    np.testing.assert_allclose(
        sp.csr_matrix(ev.evaluate_sparse_jacobian(x)).toarray(),
        traced._evaluate_dense_jacobian(x),
        rtol=1e-12,
        atol=1e-12,
    )


def test_jax_evaluator_affine_arm_refuses_parameters(monkeypatch):
    monkeypatch.setattr(nle, "_DENSE_JACOBIAN_COMPILE_LIMIT", 0)
    m = Model("p")
    x = m.continuous("x", shape=(3,), lb=0, ub=1)
    p = m.parameter("p", value=2.0)
    m.minimize(dm.sum(x))
    m.subject_to(p * x <= 1.0)
    ev = nle.NLPEvaluator(m)
    assert ev.sparsity_pattern is not None
    assert getattr(ev, "_affine_jacobian", None) is None


def test_solve_through_vectorized_rung_matches_tape_route(monkeypatch):
    base = _transport(8).solve()
    assert base.status == "optimal"

    monkeypatch.setattr(nlc, "_MAX_TAPE_NODES", 10)
    monkeypatch.setattr(nle, "_DENSE_JACOBIAN_COMPILE_LIMIT", 0)

    def _boom(*_a, **_k):
        raise AssertionError("dense autodiff extractor reached")

    monkeypatch.setattr(pc, "_extract_lp_data_autodiff", _boom)
    res = _transport(8).solve()
    assert res.status == "optimal"
    assert res.objective == pytest.approx(base.objective, rel=1e-9, abs=1e-9)
    assert res.constraint_duals is not None and base.constraint_duals is not None
    assert set(res.constraint_duals) == {"sup", "dem"} == set(base.constraint_duals)
    for k in ("sup", "dem"):
        np.testing.assert_allclose(
            res.constraint_duals[k], base.constraint_duals[k], rtol=1e-7, atol=1e-9
        )
